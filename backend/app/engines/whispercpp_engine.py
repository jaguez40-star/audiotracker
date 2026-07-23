"""Transcripcion con whisper.cpp acelerado por Vulkan.

Existe porque la RX 6700 XT (RDNA2/gfx1031) no tiene ruta CUDA ni ROCm en
Windows, pero si Vulkan. Se usa el binario precompilado de lemonade-sdk, que
evita compilar desde fuente y ademas pasa el filtro de Smart App Control (un
ejecutable compilado localmente no lo pasaria).

Ventaja secundaria nada menor con 8 GB de RAM: el modelo se carga en la VRAM del
subproceso, no en la memoria del backend. Eso libera RAM del sistema para
pyannote, que sigue en CPU y es quien la necesita.

Se comunica por subproceso en vez de por binding: whisper.cpp expone un CLI
estable con salida JSON, y un binding requeriria compilar — justo lo que se
quiere evitar.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger
from app.engines.base import ProgressCallback, TranscriptionSegment

log = get_logger(__name__)

# whisper.cpp con -pp emite lineas del tipo:
#   whisper_print_progress_callback: progress =  40%
_PROGRESO = re.compile(r"progress\s*=\s*(\d+)%")


class WhisperCppBinaryMissing(RuntimeError):
    """El binario o el modelo no estan donde dice la configuracion."""


class WhisperCppEngine:
    name = "whisper.cpp-vulkan"

    def __init__(
        self,
        binary: Path | None = None,
        model: Path | None = None,
        use_gpu: bool | None = None,
    ) -> None:
        self.binary = Path(binary or settings.whispercpp_binary)
        self.model = Path(model or settings.whispercpp_model)
        self.use_gpu = settings.whispercpp_use_gpu if use_gpu is None else use_gpu

    def disponible(self) -> tuple[bool, str]:
        """Comprueba si este motor puede usarse, sin lanzar excepcion.

        Lo usa el selector para degradar a CPU en vez de tumbar el arranque
        cuando falta el binario o el modelo.
        """
        if not self.binary.is_file():
            return False, f"falta el binario en {self.binary}"
        if not self.model.is_file():
            return False, f"falta el modelo en {self.model}"
        return True, "ok"

    def __enter__(self) -> WhisperCppEngine:
        # No hay modelo que cargar en este proceso: lo hace el subproceso. Lo
        # unico que se valida aqui es que exista lo que vamos a invocar, para
        # fallar con un mensaje claro y no con un error de subproceso opaco.
        if not self.binary.is_file():
            raise WhisperCppBinaryMissing(
                f"No existe el binario de whisper.cpp en {self.binary}. "
                "Revisa WHISPERCPP_BINARY en el .env."
            )
        if not self.model.is_file():
            raise WhisperCppBinaryMissing(
                f"No existe el modelo ggml en {self.model}. "
                "Descargalo o revisa WHISPERCPP_MODEL en el .env."
            )
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def _construir_comando(self, wav: Path, salida: Path) -> list[str]:
        cmd = [
            str(self.binary),
            "-m", str(self.model),
            "-f", str(wav),
            "-oj",                       # resultado en JSON
            "-of", str(salida),          # prefijo: genera <salida>.json
            "-pp",                       # progreso a stderr
            "-t", str(settings.whispercpp_threads),
            "-bs", str(settings.whisper_beam_size),
        ]
        if settings.whisper_language:
            cmd += ["-l", settings.whisper_language]
        else:
            cmd += ["-l", "auto"]
        if not self.use_gpu:
            cmd.append("-ng")
        return cmd

    def transcribe(
        self,
        audio_path: str,
        on_progress: ProgressCallback | None = None,
    ) -> list[TranscriptionSegment]:
        wav = Path(audio_path)

        # El JSON va a un temporal propio: whisper.cpp lo escribe junto al
        # prefijo indicado, y no queremos ensuciar el directorio de uploads.
        with tempfile.TemporaryDirectory(prefix="whispercpp_") as tmp:
            prefijo = Path(tmp) / "resultado"
            cmd = self._construir_comando(wav, prefijo)

            log.info(
                "transcribiendo con whisper.cpp",
                gpu=self.use_gpu,
                modelo=self.model.name,
            )

            proceso = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            # El progreso solo existe mientras el subproceso vive: hay que
            # consumir la salida en streaming, no esperar a que termine.
            ultimas_lineas: list[str] = []
            assert proceso.stdout is not None
            for linea in proceso.stdout:
                ultimas_lineas.append(linea.rstrip())
                del ultimas_lineas[:-40]  # solo interesan las ultimas si falla
                if on_progress:
                    m = _PROGRESO.search(linea)
                    if m:
                        on_progress(min(int(m.group(1)) / 100.0, 1.0))

            codigo = proceso.wait()
            if codigo != 0:
                cola = "\n".join(ultimas_lineas)
                raise RuntimeError(
                    f"whisper.cpp termino con codigo {codigo}.\n{cola}"
                )

            json_path = prefijo.with_suffix(".json")
            if not json_path.is_file():
                cola = "\n".join(ultimas_lineas)
                raise RuntimeError(
                    f"whisper.cpp no genero {json_path.name}.\n{cola}"
                )

            datos = json.loads(json_path.read_text(encoding="utf-8"))

        segmentos = _parsear(datos)
        if on_progress:
            on_progress(1.0)

        log.info("transcripcion completa", segmentos=len(segmentos))
        return segmentos


def _parsear(datos: dict) -> list[TranscriptionSegment]:
    """Convierte el JSON de whisper.cpp a nuestro formato.

    Los offsets vienen en milisegundos; el resto del sistema trabaja en segundos.
    """
    segmentos: list[TranscriptionSegment] = []
    for item in datos.get("transcription", []):
        texto = item.get("text", "").strip()
        if not texto:
            continue
        offsets = item.get("offsets", {})
        segmentos.append(
            TranscriptionSegment(
                start=offsets.get("from", 0) / 1000.0,
                end=offsets.get("to", 0) / 1000.0,
                text=texto,
            )
        )
    return segmentos
