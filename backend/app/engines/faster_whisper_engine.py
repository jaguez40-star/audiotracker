"""Transcripcion con faster-whisper (CTranslate2).

Corre en CPU con cuantizacion int8. En este equipo no hay alternativa: la RX
6700 XT es RDNA2 (gfx1031) y CTranslate2 solo compila kernels ROCm para
gfx1030/gfx1100+, ademas de que AMD no distribuye runtime HIP para RDNA2 en
Windows. Verificado contra el propio Ollama, que rechaza la GPU en arranque.
"""

from __future__ import annotations

import gc

from app.core.config import settings
from app.core.logging import get_logger
from app.engines.base import ProgressCallback, TranscriptionSegment

log = get_logger(__name__)


class FasterWhisperEngine:
    name = "faster-whisper"

    def __init__(
        self,
        model_size: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
    ) -> None:
        self.model_size = model_size or settings.whisper_model
        self.device = device or settings.whisper_device
        self.compute_type = compute_type or settings.whisper_compute_type
        self._model = None

    def __enter__(self) -> FasterWhisperEngine:
        # Import diferido: cargar CTranslate2 tarda y solo hace falta al procesar.
        from faster_whisper import WhisperModel

        log.info(
            "cargando modelo de transcripcion",
            model=self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        kwargs: dict[str, object] = {
            "device": self.device,
            "compute_type": self.compute_type,
        }
        if settings.whisper_cpu_threads > 0:
            kwargs["cpu_threads"] = settings.whisper_cpu_threads

        self._model = WhisperModel(self.model_size, **kwargs)
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._model = None
        gc.collect()
        log.info("modelo de transcripcion liberado")

    def transcribe(
        self,
        audio_path: str,
        on_progress: ProgressCallback | None = None,
    ) -> list[TranscriptionSegment]:
        if self._model is None:
            raise RuntimeError("El motor debe usarse como context manager")

        segments_iter, info = self._model.transcribe(
            audio_path,
            language=settings.whisper_language,
            beam_size=settings.whisper_beam_size,
            vad_filter=True,
        )

        total = info.duration or 0.0
        log.info(
            "transcribiendo",
            duracion_s=round(total, 1),
            idioma=info.language,
        )

        # faster-whisper devuelve un generador perezoso: el trabajo real ocurre
        # al iterarlo, por eso el progreso se reporta aqui y no antes.
        results: list[TranscriptionSegment] = []
        for seg in segments_iter:
            text = seg.text.strip()
            if text:
                results.append(
                    TranscriptionSegment(start=seg.start, end=seg.end, text=text)
                )
            if on_progress and total > 0:
                on_progress(min(seg.end / total, 1.0))

        if on_progress:
            on_progress(1.0)

        log.info("transcripcion completa", segmentos=len(results))
        return results
