"""Pruebas del motor whisper.cpp.

No ejecutan el binario: cubren la construccion del comando, el parseo de su
salida JSON y la deteccion de disponibilidad.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.engines.whispercpp_engine import (
    WhisperCppBinaryMissing,
    WhisperCppEngine,
    _parsear,
)


@pytest.fixture
def falsos(tmp_path: Path) -> tuple[Path, Path]:
    """Binario y modelo simulados, solo para que existan en disco."""
    binario = tmp_path / "whisper-cli.exe"
    modelo = tmp_path / "ggml-large-v3-q5_0.bin"
    binario.write_bytes(b"MZ")
    modelo.write_bytes(b"ggml")
    return binario, modelo


class TestDisponibilidad:
    def test_ok_cuando_ambos_existen(self, falsos):
        binario, modelo = falsos
        ok, motivo = WhisperCppEngine(binary=binario, model=modelo).disponible()

        assert ok
        assert motivo == "ok"

    def test_detecta_binario_ausente(self, tmp_path: Path, falsos):
        _, modelo = falsos
        ok, motivo = WhisperCppEngine(
            binary=tmp_path / "no-existe.exe", model=modelo
        ).disponible()

        assert not ok
        assert "binario" in motivo

    def test_detecta_modelo_ausente(self, tmp_path: Path, falsos):
        binario, _ = falsos
        ok, motivo = WhisperCppEngine(
            binary=binario, model=tmp_path / "no-existe.bin"
        ).disponible()

        assert not ok
        assert "modelo" in motivo

    def test_entrar_sin_binario_da_error_claro(self, tmp_path: Path, falsos):
        _, modelo = falsos
        motor = WhisperCppEngine(binary=tmp_path / "nada.exe", model=modelo)

        with pytest.raises(WhisperCppBinaryMissing, match="binario"):
            motor.__enter__()


class TestComando:
    def test_gpu_no_agrega_flag_de_cpu(self, falsos, tmp_path: Path):
        binario, modelo = falsos
        motor = WhisperCppEngine(binary=binario, model=modelo, use_gpu=True)

        cmd = motor._construir_comando(tmp_path / "a.wav", tmp_path / "out")

        assert "-ng" not in cmd

    def test_cpu_agrega_flag_ng(self, falsos, tmp_path: Path):
        binario, modelo = falsos
        motor = WhisperCppEngine(binary=binario, model=modelo, use_gpu=False)

        cmd = motor._construir_comando(tmp_path / "a.wav", tmp_path / "out")

        assert "-ng" in cmd

    def test_incluye_json_y_progreso(self, falsos, tmp_path: Path):
        binario, modelo = falsos
        motor = WhisperCppEngine(binary=binario, model=modelo)

        cmd = motor._construir_comando(tmp_path / "a.wav", tmp_path / "out")

        # Sin -oj no hay salida que parsear; sin -pp no hay progreso que reportar.
        assert "-oj" in cmd
        assert "-pp" in cmd
        assert "-m" in cmd and str(modelo) in cmd
        assert "-f" in cmd


class TestParseo:
    def test_convierte_milisegundos_a_segundos(self):
        datos = {
            "transcription": [
                {"offsets": {"from": 0, "to": 3200}, "text": " Hola."},
                {"offsets": {"from": 3500, "to": 7000}, "text": " Que tal."},
            ]
        }

        segmentos = _parsear(datos)

        assert len(segmentos) == 2
        assert segmentos[0].start == 0.0
        assert segmentos[0].end == 3.2
        assert segmentos[1].start == 3.5
        assert segmentos[1].end == 7.0

    def test_recorta_espacios(self):
        # whisper.cpp antepone un espacio a casi todos los segmentos.
        datos = {"transcription": [{"offsets": {"from": 0, "to": 1000}, "text": "  Hola  "}]}

        assert _parsear(datos)[0].text == "Hola"

    def test_descarta_segmentos_vacios(self):
        datos = {
            "transcription": [
                {"offsets": {"from": 0, "to": 1000}, "text": "   "},
                {"offsets": {"from": 1000, "to": 2000}, "text": " Real."},
            ]
        }

        segmentos = _parsear(datos)

        assert len(segmentos) == 1
        assert segmentos[0].text == "Real."

    def test_json_sin_transcripcion(self):
        assert _parsear({}) == []

    def test_tolera_offsets_ausentes(self):
        datos = {"transcription": [{"text": " Sin tiempos."}]}

        segmentos = _parsear(datos)

        assert len(segmentos) == 1
        assert segmentos[0].start == 0.0
