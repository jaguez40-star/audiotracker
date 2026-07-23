"""Registro de trabajos en memoria.

Es una herramienta local de un solo usuario, asi que no hay base de datos: el
estado vive en un dict y se pierde al reiniciar. El transcript final si queda en
disco, que es lo unico que importa conservar.

El acceso va protegido por lock porque el procesamiento corre en un hilo del
threadpool de FastAPI mientras el endpoint de estado lee desde el hilo del
event loop.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

from app.engines.base import SpeakerSegment


class JobStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


class JobStage(StrEnum):
    PENDING = "pending"
    PREPARING = "preparing"
    TRANSCRIBING = "transcribing"
    DIARIZING = "diarizing"
    MERGING = "merging"
    FINISHED = "finished"


# Peso de cada etapa en la barra de progreso global. La diarizacion domina el
# tiempo total por un margen amplio en CPU, y reflejarlo evita que la barra se
# quede clavada en el mismo numero durante la mayor parte del proceso.
STAGE_WEIGHTS: dict[JobStage, tuple[float, float]] = {
    JobStage.PREPARING: (0.00, 0.03),
    JobStage.TRANSCRIBING: (0.03, 0.28),
    JobStage.DIARIZING: (0.28, 0.97),
    JobStage.MERGING: (0.97, 1.00),
}


@dataclass
class Job:
    id: str
    filename: str
    status: JobStatus = JobStatus.QUEUED
    stage: JobStage = JobStage.PENDING
    progress: float = 0.0
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    duration_seconds: float | None = None
    speaker_count: int | None = None
    segments: list[SpeakerSegment] = field(default_factory=list)
    text_path: Path | None = None

    @property
    def processing_seconds(self) -> float | None:
        """Segundos de computo. Es el dato que dice si el equipo da la talla."""
        if self.started_at is None:
            return None
        fin = self.finished_at or datetime.now(timezone.utc)
        return (fin - self.started_at).total_seconds()

    @property
    def speed_ratio(self) -> float | None:
        """Cuantos segundos de computo por segundo de audio.

        Es la metrica comparable entre archivos: 3.0 significa que procesar un
        audio cuesta el triple de su duracion.
        """
        gastado = self.processing_seconds
        if gastado is None or not self.duration_seconds:
            return None
        return gastado / self.duration_seconds


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, filename: str) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], filename=filename)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_all(self) -> list[Job]:
        with self._lock:
            return sorted(
                self._jobs.values(), key=lambda j: j.created_at, reverse=True
            )

    def update(self, job_id: str, **fields: object) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in fields.items():
                setattr(job, key, value)

    def set_stage_progress(self, job_id: str, stage: JobStage, fraction: float) -> None:
        """Traduce el avance dentro de una etapa a la barra global."""
        floor, ceiling = STAGE_WEIGHTS.get(stage, (0.0, 1.0))
        overall = floor + (ceiling - floor) * min(max(fraction, 0.0), 1.0)
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.stage = stage
            # Monotono: reintentos o hooks desordenados no deben hacerla retroceder.
            job.progress = max(job.progress, round(overall, 4))


store = JobStore()
