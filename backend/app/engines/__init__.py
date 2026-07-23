"""Seleccion del motor de transcripcion.

La eleccion se resuelve aqui y no en el pipeline, para que el resto del codigo
solo conozca el protocolo TranscriptionEngine.
"""

from __future__ import annotations

from app.core.config import settings
from app.core.logging import get_logger
from app.engines.base import TranscriptionEngine
from app.engines.faster_whisper_engine import FasterWhisperEngine
from app.engines.whispercpp_engine import WhisperCppEngine

log = get_logger(__name__)


def build_transcription_engine() -> TranscriptionEngine:
    """Devuelve el motor configurado, con degradacion a CPU si no es viable.

    Preferimos arrancar mas lento antes que no arrancar: si el binario de
    whisper.cpp o el modelo ggml no estan donde dice la configuracion, se avisa
    y se usa faster-whisper en CPU.
    """
    if settings.transcription_engine == "whispercpp":
        motor = WhisperCppEngine()
        ok, motivo = motor.disponible()
        if ok:
            return motor
        log.warning(
            "whisper.cpp no disponible, se usara CPU",
            motivo=motivo,
        )

    return FasterWhisperEngine()


def describe_transcription_engine() -> dict[str, object]:
    """Resumen del motor activo, para /api/health."""
    if settings.transcription_engine == "whispercpp":
        motor = WhisperCppEngine()
        ok, motivo = motor.disponible()
        if ok:
            return {
                "engine": motor.name,
                "device": "gpu (vulkan)" if motor.use_gpu else "cpu",
                "model": motor.model.name,
            }
        return {
            "engine": FasterWhisperEngine.name,
            "device": "cpu",
            "model": settings.whisper_model,
            "fallback_reason": motivo,
        }

    return {
        "engine": FasterWhisperEngine.name,
        "device": settings.whisper_device,
        "model": settings.whisper_model,
    }


__all__ = [
    "TranscriptionEngine",
    "FasterWhisperEngine",
    "WhisperCppEngine",
    "build_transcription_engine",
    "describe_transcription_engine",
]
