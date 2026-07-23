"""Normalizacion de audio a un formato que ambos motores digieren sin sorpresas.

Los dos motores decodifican por su cuenta, pero con backends distintos:
faster-whisper usa PyAV y pyannote usa torchaudio/soundfile, que es bastante mas
limitado con contenedores tipo mp4/m4a. Convertir una sola vez a WAV PCM 16 kHz
mono elimina esa clase de fallo y ademas evita decodificar el archivo dos veces.

Usamos el decodificador de faster-whisper (PyAV, que trae las librerias de
FFmpeg embebidas) en vez del binario ffmpeg del sistema, para no arrastrar una
dependencia externa que habria que instalar aparte.
"""

from __future__ import annotations

from pathlib import Path

from app.core.logging import get_logger

log = get_logger(__name__)

TARGET_SAMPLE_RATE = 16_000


def to_wav_16k_mono(source: Path, dest: Path) -> Path:
    """Convierte cualquier formato soportado a WAV PCM 16 kHz mono."""
    import soundfile as sf
    from faster_whisper.audio import decode_audio

    log.info("normalizando audio", origen=source.name)

    samples = decode_audio(str(source), sampling_rate=TARGET_SAMPLE_RATE)

    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), samples, TARGET_SAMPLE_RATE, subtype="PCM_16")

    duracion = len(samples) / TARGET_SAMPLE_RATE
    log.info("audio normalizado", duracion_s=round(duracion, 1), destino=dest.name)
    return dest


def probe_duration(path: Path) -> float:
    """Duracion en segundos, sin cargar el audio completo en memoria."""
    import soundfile as sf

    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)
