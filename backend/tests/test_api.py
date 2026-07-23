"""Smoke tests de la API.

No cargan modelos: solo verifican contratos de endpoints, validacion de entrada
y manejo de errores.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_responde(client: TestClient):
    res = client.get("/api/health")

    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert "whisper_model" in body
    assert "hf_token_configured" in body


def test_openapi_declara_los_endpoints(client: TestClient):
    paths = client.get("/openapi.json").json()["paths"]

    assert "/api/transcribe" in paths
    assert "/api/jobs/{job_id}" in paths
    assert "/api/jobs/{job_id}/download" in paths


def test_job_inexistente_da_404(client: TestClient):
    assert client.get("/api/jobs/noexiste").status_code == 404
    assert client.get("/api/jobs/noexiste/result").status_code == 404
    assert client.get("/api/jobs/noexiste/download").status_code == 404


def test_lista_de_jobs(client: TestClient):
    res = client.get("/api/jobs")

    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_rechaza_extension_no_soportada(client: TestClient):
    res = client.post(
        "/api/transcribe",
        files={"file": ("documento.pdf", io.BytesIO(b"no soy audio"), "application/pdf")},
    )

    assert res.status_code == 400
    assert ".pdf" in res.json()["detail"]


def test_sin_archivo_da_422(client: TestClient):
    assert client.post("/api/transcribe").status_code == 422


class TestNumSpeakers:
    """El numero de hablantes viaja por peticion, no por configuracion global."""

    def test_declarado_en_el_esquema_como_opcional(self, client: TestClient):
        esquema = client.get("/openapi.json").json()
        cuerpo = esquema["paths"]["/api/transcribe"]["post"]["requestBody"]
        campos = cuerpo["content"]["multipart/form-data"]["schema"]["$ref"]
        nombre = campos.rsplit("/", 1)[-1]
        props = esquema["components"]["schemas"][nombre]["properties"]

        assert "num_speakers" in props
        assert "num_speakers" not in esquema["components"]["schemas"][nombre].get(
            "required", []
        )

    @pytest.mark.parametrize("valor", [0, 21, -3])
    def test_rechaza_valores_fuera_de_rango(self, client: TestClient, valor: int):
        res = client.post(
            "/api/transcribe",
            files={"file": ("a.wav", io.BytesIO(b"x"), "audio/wav")},
            data={"num_speakers": str(valor)},
        )

        assert res.status_code == 422

    def test_la_extension_se_valida_antes_que_el_token(self, client: TestClient):
        # Un .pdf debe dar 400 aunque num_speakers sea valido: no tiene sentido
        # quejarse del token por un archivo que igual se iba a rechazar.
        res = client.post(
            "/api/transcribe",
            files={"file": ("a.pdf", io.BytesIO(b"x"), "application/pdf")},
            data={"num_speakers": "2"},
        )

        assert res.status_code == 400
