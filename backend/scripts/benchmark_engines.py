"""Compara motores de transcripcion sobre el mismo audio.

    venv\\Scripts\\python scripts\\benchmark_engines.py <audio> [--skip-cpu]

Mide whisper.cpp en GPU (Vulkan), whisper.cpp en CPU y faster-whisper en CPU.
Los dos primeros usan el mismo binario y el mismo modelo, asi que la unica
variable es el dispositivo: la comparacion es limpia.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.audio import probe_duration, to_wav_16k_mono  # noqa: E402
from app.engines.faster_whisper_engine import FasterWhisperEngine  # noqa: E402
from app.engines.whispercpp_engine import WhisperCppEngine  # noqa: E402


def medir(nombre: str, motor, wav: str, duracion: float) -> dict | None:
    print(f"\n{'=' * 64}\n{nombre}\n{'=' * 64}")
    try:
        inicio = time.perf_counter()
        with motor as m:
            segmentos = m.transcribe(wav)
        transcurrido = time.perf_counter() - inicio
    except Exception as exc:
        print(f"  FALLO: {type(exc).__name__}: {str(exc)[:300]}")
        return None

    ratio = transcurrido / duracion if duracion else 0.0
    texto = " ".join(s.text for s in segmentos)
    print(f"  tiempo    : {transcurrido:.1f} s")
    print(f"  ratio     : {ratio:.2f}x  ({1 / ratio:.1f}x tiempo real)" if ratio else "")
    print(f"  segmentos : {len(segmentos)}")
    print(f"  texto     : {texto[:200]}{'...' if len(texto) > 200 else ''}")
    return {"nombre": nombre, "s": transcurrido, "ratio": ratio, "texto": texto}


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    origen = Path(sys.argv[1]).resolve()
    if not origen.is_file():
        print(f"No existe: {origen}")
        return 1

    saltar_cpu = "--skip-cpu" in sys.argv

    wav = origen.with_name(f"{origen.stem}_bench16k.wav")
    to_wav_16k_mono(origen, wav)
    duracion = probe_duration(wav)
    print(f"\naudio: {origen.name}  ({duracion:.1f} s)")

    resultados = []
    r = medir(
        "whisper.cpp  ·  GPU (Vulkan)",
        WhisperCppEngine(use_gpu=True),
        str(wav),
        duracion,
    )
    if r:
        resultados.append(r)

    if not saltar_cpu:
        r = medir(
            "whisper.cpp  ·  CPU (mismo modelo)",
            WhisperCppEngine(use_gpu=False),
            str(wav),
            duracion,
        )
        if r:
            resultados.append(r)

        r = medir(
            "faster-whisper  ·  CPU (modelo small)",
            FasterWhisperEngine(),
            str(wav),
            duracion,
        )
        if r:
            resultados.append(r)

    wav.unlink(missing_ok=True)

    if len(resultados) > 1:
        print(f"\n{'=' * 64}\nRESUMEN\n{'=' * 64}")
        base = max(x["s"] for x in resultados)
        for x in sorted(resultados, key=lambda d: d["s"]):
            print(
                f"  {x['nombre']:<40} {x['s']:>7.1f} s  "
                f"{x['ratio']:>5.2f}x  ({base / x['s']:.1f}x mas rapido que el peor)"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
