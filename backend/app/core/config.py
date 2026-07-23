"""Configuracion central. Todo ajustable por .env sin tocar codigo."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BeforeValidator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


def _blank_to_none(value: Any) -> Any:
    """Trata una variable vacia en el .env como ausente.

    `MIN_SPEAKERS=` es la forma natural de escribir "sin valor", pero pydantic lo
    recibe como cadena vacia e intenta parsearla como entero. Sin esto, el
    backend no arranca con el .env.example tal cual se entrega.
    """
    return None if isinstance(value, str) and not value.strip() else value


OptionalInt = Annotated[int | None, BeforeValidator(_blank_to_none)]
OptionalStr = Annotated[str | None, BeforeValidator(_blank_to_none)]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Servidor ---
    host: str = "127.0.0.1"
    port: int = 6024
    cors_origins: list[str] = ["http://localhost:6023", "http://127.0.0.1:6023"]

    # --- Almacenamiento ---
    upload_dir: Path = BASE_DIR / "data" / "uploads"
    output_dir: Path = BASE_DIR / "data" / "outputs"
    keep_uploads: bool = False

    # --- Seleccion de motor de transcripcion ---
    # "whispercpp" usa la GPU via Vulkan; "faster-whisper" corre en CPU.
    # El fallback a CPU es automatico si el binario o el modelo no estan.
    transcription_engine: Literal["whispercpp", "faster-whisper"] = "whispercpp"

    # --- whisper.cpp (GPU via Vulkan) ---
    # La RX 6700 XT es RDNA2/gfx1031: sin CUDA ni ROCm en Windows, pero Vulkan
    # si funciona. Binario precompilado de lemonade-sdk, que ademas pasa el
    # filtro de Smart App Control.
    whispercpp_binary: Path = BASE_DIR.parent / "vendor" / "whisper-vulkan" / "whisper-cli.exe"
    whispercpp_model: Path = BASE_DIR.parent / "models" / "ggml-large-v3-q5_0.bin"
    whispercpp_use_gpu: bool = True
    whispercpp_threads: int = 6

    # --- faster-whisper (CPU) ---
    whisper_model: str = "small"
    whisper_device: Literal["cpu", "cuda"] = "cpu"
    whisper_compute_type: str = "int8"
    whisper_language: OptionalStr = "es"
    whisper_beam_size: int = 5
    # 0 = dejar que CTranslate2 decida. El i5-10400 tiene 6 nucleos fisicos.
    whisper_cpu_threads: int = 0

    # --- Motor de diarizacion ---
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    # pydantic-settings mapea hf_token <- HF_TOKEN sin necesidad de alias.
    hf_token: OptionalStr = None
    # None = pyannote decide cuantos hablantes hay
    min_speakers: OptionalInt = None
    max_speakers: OptionalInt = None

    # --- Limites ---
    max_upload_mb: int = 500

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


settings = Settings()

settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.output_dir.mkdir(parents=True, exist_ok=True)
