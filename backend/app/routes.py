"""Endpoints REST."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.logging import get_logger
from app.engines import describe_transcription_engine
from app.jobs import JobStatus, store
from app.pipeline import process_audio
from app.schemas import HealthOut, JobOut, JobResultOut

log = get_logger(__name__)
router = APIRouter(prefix="/api")

# Formatos que PyAV decodifica sin problema. Mas permisivo que mp4/wav porque no
# cuesta nada y evita rechazar grabaciones de celular o de reuniones.
EXTENSIONES_VALIDAS = {
    ".mp4", ".wav", ".mp3", ".m4a", ".aac",
    ".flac", ".ogg", ".opus", ".webm", ".mkv", ".mov",
}

CHUNK = 1024 * 1024


@router.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    motor = describe_transcription_engine()
    return HealthOut(
        status="ok",
        engine=str(motor["engine"]),
        device=str(motor["device"]),
        whisper_model=str(motor["model"]),
        diarization_model=settings.diarization_model,
        hf_token_configured=bool(settings.hf_token),
        fallback_reason=motor.get("fallback_reason"),  # type: ignore[arg-type]
    )


@router.post("/transcribe", response_model=JobOut, status_code=status.HTTP_202_ACCEPTED)
async def transcribe(
    file: UploadFile,
    background: BackgroundTasks,
    num_speakers: Annotated[
        int | None,
        Form(
            ge=1,
            le=20,
            description=(
                "Cuantas personas hablan, si se sabe. Omitir para deteccion "
                "automatica. Indicarlo reduce mucho los falsos hablantes."
            ),
        ),
    ] = None,
) -> JobOut:
    if not file.filename:
        raise HTTPException(400, "Archivo sin nombre")

    extension = Path(file.filename).suffix.lower()
    if extension not in EXTENSIONES_VALIDAS:
        raise HTTPException(
            400,
            f"Formato '{extension}' no soportado. "
            f"Validos: {', '.join(sorted(EXTENSIONES_VALIDAS))}",
        )

    if not settings.hf_token:
        raise HTTPException(
            503,
            "Falta HF_TOKEN en el .env del backend. Sin el no se puede diarizar. "
            "Ver README.",
        )

    job = store.create(file.filename)
    destino = settings.upload_dir / f"{job.id}{extension}"

    # Se escribe por bloques para no cargar en RAM un archivo que puede pesar
    # cientos de MB, y se corta apenas supera el limite.
    escrito = 0
    try:
        with destino.open("wb") as salida:
            while bloque := await file.read(CHUNK):
                escrito += len(bloque)
                if escrito > settings.max_upload_bytes:
                    salida.close()
                    destino.unlink(missing_ok=True)
                    raise HTTPException(
                        413, f"El archivo supera el limite de {settings.max_upload_mb} MB"
                    )
                salida.write(bloque)
    except HTTPException:
        raise
    except OSError as exc:
        destino.unlink(missing_ok=True)
        raise HTTPException(500, f"No se pudo guardar el archivo: {exc}") from exc

    log.info(
        "archivo recibido",
        job_id=job.id,
        nombre=file.filename,
        bytes=escrito,
        num_speakers=num_speakers,
    )
    background.add_task(process_audio, job.id, destino, num_speakers)
    return JobOut.from_job(job)


@router.get("/jobs", response_model=list[JobOut])
def list_jobs() -> list[JobOut]:
    return [JobOut.from_job(j) for j in store.list_all()]


@router.get("/jobs/{job_id}", response_model=JobOut)
def job_status(job_id: str) -> JobOut:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job no encontrado")
    return JobOut.from_job(job)


@router.get("/jobs/{job_id}/result", response_model=JobResultOut)
def job_result(job_id: str) -> JobResultOut:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job no encontrado")
    if job.status != JobStatus.DONE:
        raise HTTPException(409, f"El job aun no termina (estado: {job.status})")

    texto = job.text_path.read_text(encoding="utf-8") if job.text_path else ""
    return JobResultOut.from_job_with_result(job, texto)


@router.get("/jobs/{job_id}/download")
def download(job_id: str) -> FileResponse:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job no encontrado")
    if job.status != JobStatus.DONE or job.text_path is None:
        raise HTTPException(409, "El transcript aun no esta listo")
    if not job.text_path.exists():
        raise HTTPException(410, "El archivo de transcript ya no existe en disco")

    nombre = f"{Path(job.filename).stem}_transcript.txt"
    return FileResponse(job.text_path, media_type="text/plain; charset=utf-8", filename=nombre)
