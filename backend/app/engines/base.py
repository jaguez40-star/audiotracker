"""Contratos de los motores de audio.

El objetivo de esta capa es que la eleccion de motor sea un detalle reemplazable.
Hoy corremos faster-whisper + pyannote en CPU porque esta maquina (AMD RDNA2 en
Windows) no tiene ruta GPU viable, pero eso puede cambiar: whisper.cpp con
Vulkan, o sherpa-onnx via DirectML. Cuando pase, se escribe un motor nuevo que
cumpla estos protocolos y no se toca nada mas.

Contrato de memoria: los motores se usan como context manager y liberan el
modelo al salir. El pipeline nunca mantiene dos modelos cargados a la vez, que
es lo que hace viable correr esto en 8 GB de RAM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

# Reporta avance 0.0..1.0 dentro de la etapa actual.
ProgressCallback = Callable[[float], None]


@dataclass(frozen=True, slots=True)
class TranscriptionSegment:
    """Un fragmento de texto con sus tiempos, sin atribuir hablante."""

    start: float
    end: float
    text: str


@dataclass(frozen=True, slots=True)
class SpeakerTurn:
    """Un intervalo en el que habla una sola persona."""

    start: float
    end: float
    speaker: str


@dataclass(frozen=True, slots=True)
class SpeakerSegment:
    """Resultado final: texto con hablante atribuido."""

    start: float
    end: float
    speaker: str
    text: str


@runtime_checkable
class TranscriptionEngine(Protocol):
    """Audio -> segmentos de texto con tiempos."""

    name: str

    def __enter__(self) -> TranscriptionEngine: ...

    def __exit__(self, *exc_info: object) -> None: ...

    def transcribe(
        self,
        audio_path: str,
        on_progress: ProgressCallback | None = None,
    ) -> list[TranscriptionSegment]: ...


@runtime_checkable
class DiarizationEngine(Protocol):
    """Audio -> turnos de habla por persona."""

    name: str

    def __enter__(self) -> DiarizationEngine: ...

    def __exit__(self, *exc_info: object) -> None: ...

    def diarize(
        self,
        audio_path: str,
        on_progress: ProgressCallback | None = None,
        num_speakers: int | None = None,
    ) -> list[SpeakerTurn]: ...
