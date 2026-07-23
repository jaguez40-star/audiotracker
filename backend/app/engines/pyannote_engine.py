"""Diarizacion con pyannote.audio (modelo speaker-diarization-3.1).

Requiere token de HuggingFace y aceptar la licencia del modelo una vez. Ver
README.

Corre en CPU: pyannote es PyTorch, y PyTorch no tiene build ROCm para Windows
que cubra RDNA2. Es la etapa mas lenta del pipeline por un margen amplio; si los
tiempos molestan, el reemplazo natural es sherpa-onnx (ONNX Runtime, bastante
mas liviano) implementando el mismo protocolo DiarizationEngine.

Nota sobre decodificacion: pyannote 4 delega la lectura de audio en torchcodec,
que aqui no carga porque la politica WDAC del equipo bloquea sus DLLs sin firmar
(WinError 4551). En vez de pelear con eso, le pasamos el audio ya decodificado en
memoria como {'waveform', 'sample_rate'}, que es la via que el propio pyannote
recomienda y ademas evita releer el archivo del disco.
"""

from __future__ import annotations

import gc

from app.core.config import settings
from app.core.logging import get_logger
from app.engines.base import ProgressCallback, SpeakerTurn

log = get_logger(__name__)


class _ProgressHook:
    """Traduce los avisos de pyannote a una fraccion 0..1.

    pyannote ejecuta varias etapas internas y reporta avance dentro de cada una.
    Sin saber cuantas etapas habra de antemano, informar el avance crudo haria
    que la barra volviera atras en cada etapa nueva. Repartimos el rango de
    forma monotona: cada etapa nueva arranca donde termino la anterior.
    """

    _STAGE_SHARE = 0.45

    def __init__(self, on_progress: ProgressCallback | None) -> None:
        self._on_progress = on_progress
        self._floor = 0.0
        self._ceiling = self._STAGE_SHARE
        self._stage: str | None = None

    def __call__(
        self,
        step_name: str,
        step_artifact: object = None,
        file: object = None,
        total: int | None = None,
        completed: int | None = None,
    ) -> None:
        if self._on_progress is None:
            return

        if step_name != self._stage:
            self._stage = step_name
            self._floor = self._ceiling
            # Cada etapa consume una fraccion del rango restante, de modo que el
            # avance siempre crece y nunca llega a 1.0 antes de tiempo.
            self._ceiling = self._floor + (1.0 - self._floor) * self._STAGE_SHARE

        if total and completed is not None and total > 0:
            frac = min(completed / total, 1.0)
            self._on_progress(self._floor + (self._ceiling - self._floor) * frac)
        else:
            self._on_progress(self._floor)


def _as_annotation(resultado: object):
    """Extrae el Annotation del resultado del pipeline.

    pyannote 3.x devolvia un Annotation directo; 4.x lo envuelve en un
    DiarizeOutput con tres campos. La firma declarada de `apply()` es
    `DiarizeOutput | Annotation`, o sea que ambas formas siguen siendo posibles
    segun configuracion — por eso se resuelve por duck typing y no por version.

    Usamos `speaker_diarization` (el canonico) y no `exclusive_...`: conserva los
    solapamientos, y el cruce en merge.py ya resuelve el habla simultanea
    acumulando solape por hablante y quedandose con el dominante.
    """
    interno = getattr(resultado, "speaker_diarization", None)
    if interno is not None:
        return interno
    if hasattr(resultado, "itertracks"):
        return resultado
    raise TypeError(
        f"El pipeline devolvio {type(resultado).__name__}, del que no se puede "
        "extraer un Annotation. Revisar la version de pyannote.audio."
    )


class PyannoteEngine:
    name = "pyannote-3.1"

    def __init__(self, model: str | None = None, token: str | None = None) -> None:
        self.model = model or settings.diarization_model
        self.token = token or settings.hf_token
        self._pipeline = None

    def __enter__(self) -> PyannoteEngine:
        from pyannote.audio import Pipeline

        if not self.token:
            raise RuntimeError(
                "Falta HF_TOKEN. Genera un token en "
                "https://huggingface.co/settings/tokens y acepta la licencia de "
                f"{self.model}. Ver README."
            )

        log.info("cargando modelo de diarizacion", model=self.model)
        self._pipeline = Pipeline.from_pretrained(self.model, token=self.token)

        if self._pipeline is None:
            raise RuntimeError(
                f"pyannote devolvio None al cargar {self.model}. Casi siempre "
                "significa que el token es invalido o que falta aceptar la "
                "licencia del modelo en HuggingFace. Ver README."
            )
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._pipeline = None
        gc.collect()
        log.info("modelo de diarizacion liberado")

    @staticmethod
    def _load_waveform(audio_path: str):
        """Lee el WAV a un tensor (canal, muestra) que pyannote acepta directo.

        soundfile usa libsndfile, que no pasa por torchcodec ni por las DLLs que
        WDAC bloquea.
        """
        import soundfile as sf
        import torch

        samples, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        # soundfile entrega (muestras, canales); pyannote espera (canales, muestras).
        waveform = torch.from_numpy(samples.T.copy())
        return waveform, sample_rate

    def diarize(
        self,
        audio_path: str,
        on_progress: ProgressCallback | None = None,
        num_speakers: int | None = None,
    ) -> list[SpeakerTurn]:
        if self._pipeline is None:
            raise RuntimeError("El motor debe usarse como context manager")

        # Saber cuantas voces hay elimina la parte mas fragil del problema:
        # decidir el numero de clusters. Sin esa pista, una pausa larga o un
        # cambio de entonacion bastan para que pyannote invente un hablante.
        kwargs: dict[str, int] = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        else:
            # Los valores del .env son el respaldo cuando no llega dato por peticion.
            if settings.min_speakers is not None:
                kwargs["min_speakers"] = settings.min_speakers
            if settings.max_speakers is not None:
                kwargs["max_speakers"] = settings.max_speakers

        waveform, sample_rate = self._load_waveform(audio_path)

        log.info("diarizando", muestras=int(waveform.shape[-1]), sr=sample_rate, **kwargs)
        resultado = self._pipeline(
            {"waveform": waveform, "sample_rate": sample_rate},
            hook=_ProgressHook(on_progress),
            **kwargs,
        )

        annotation = _as_annotation(resultado)
        turns = [
            SpeakerTurn(start=segment.start, end=segment.end, speaker=label)
            for segment, _, label in annotation.itertracks(yield_label=True)
        ]

        if on_progress:
            on_progress(1.0)

        hablantes = len({t.speaker for t in turns})
        log.info("diarizacion completa", turnos=len(turns), hablantes=hablantes)
        return turns
