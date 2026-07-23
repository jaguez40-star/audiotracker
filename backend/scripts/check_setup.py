"""Preflight del entorno: comprueba que todo este listo antes de procesar audio.

    venv\\Scripts\\python scripts\\check_setup.py

Verifica configuracion, runtime y acceso real a los modelos con licencia. La
comprobacion de acceso descarga un archivo pequeno de cada repositorio, porque
consultar los metadatos NO prueba nada: en un repo gated son publicos aunque los
archivos esten bloqueados.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402

OK = "  [OK]   "
FAIL = "  [FALLA]"

# pyannote 4.x necesita los tres: el modelo, su segmentador, y community-1, de
# donde saca el componente PLDA aunque se use speaker-diarization-3.1.
REPOS_REQUERIDOS = (
    ("pyannote/segmentation-3.0", "config.yaml"),
    ("pyannote/speaker-diarization-3.1", "config.yaml"),
    ("pyannote/speaker-diarization-community-1", "config.yaml"),
)


def main() -> int:
    problemas: list[str] = []

    print("\n=== Configuracion ===")
    if settings.hf_token:
        t = settings.hf_token
        print(f"{OK} HF_TOKEN presente ({t[:6]}...{t[-4:]})")
    else:
        print(f"{FAIL} HF_TOKEN ausente en backend/.env")
        problemas.append("Falta HF_TOKEN. Copia .env.example a .env y completalo.")
    print(f"         motor          : {settings.transcription_engine}")
    print(f"         modelo diariz. : {settings.diarization_model}")

    print("\n=== Motor de transcripcion ===")
    from app.engines import describe_transcription_engine
    from app.engines.whispercpp_engine import WhisperCppEngine

    if settings.transcription_engine == "whispercpp":
        motor = WhisperCppEngine()
        ok, motivo = motor.disponible()
        if ok:
            print(f"{OK} whisper.cpp    : {motor.binary.name}")
            print(f"{OK} modelo ggml    : {motor.model.name} "
                  f"({motor.model.stat().st_size / 1024**3:.2f} GB)")
            print(f"         dispositivo    : {'GPU (Vulkan)' if motor.use_gpu else 'CPU'}")
        else:
            print(f"{FAIL} whisper.cpp no disponible: {motivo}")
            print("         -> degradara a faster-whisper en CPU")
            problemas.append(f"whisper.cpp: {motivo}")

    activo = describe_transcription_engine()
    print(f"         activo         : {activo['engine']} sobre {activo['device']}")

    print("\n=== Runtime ===")
    import torch

    print(f"         torch          : {torch.__version__} (usado por pyannote, siempre CPU)")

    import ctranslate2

    tipos = ctranslate2.get_supported_compute_types("cpu")
    if settings.whisper_compute_type in tipos:
        print(f"{OK} respaldo CPU   : faster-whisper '{settings.whisper_compute_type}' ok")
    else:
        print(f"{FAIL} compute_type '{settings.whisper_compute_type}' no soportado")
        problemas.append(f"WHISPER_COMPUTE_TYPE invalido. Validos: {sorted(tipos)}")

    print("\n=== Decodificacion de audio ===")
    try:
        from faster_whisper.audio import decode_audio  # noqa: F401
        import soundfile  # noqa: F401

        print(f"{OK} PyAV + libsndfile operativos")
    except Exception as exc:
        print(f"{FAIL} {type(exc).__name__}: {exc}")
        problemas.append("Fallo la cadena de decodificacion de audio.")

    print("\n=== Acceso a modelos con licencia ===")
    if not settings.hf_token:
        print("         (omitido: sin token)")
    else:
        from huggingface_hub import hf_hub_download

        for repo, archivo in REPOS_REQUERIDOS:
            try:
                hf_hub_download(repo, archivo, token=settings.hf_token)
                print(f"{OK} {repo}")
            except Exception as exc:
                print(f"{FAIL} {repo} -> {type(exc).__name__}")
                problemas.append(
                    f"Acepta las condiciones en https://huggingface.co/{repo}"
                )

    print()
    if problemas:
        print("=" * 62)
        print("PENDIENTE:")
        for p in problemas:
            print(f"  - {p}")
        print("=" * 62)
        return 1

    print("=" * 62)
    print("TODO LISTO. El backend puede procesar audio.")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
