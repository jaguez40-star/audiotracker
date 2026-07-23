"""Modelos de respuesta de la API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.jobs import Job, JobStage, JobStatus


class SegmentOut(BaseModel):
    start: float
    end: float
    speaker: str
    text: str


class JobOut(BaseModel):
    id: str
    filename: str
    status: JobStatus
    stage: JobStage
    progress: float = Field(ge=0.0, le=1.0)
    created_at: datetime
    error: str | None = None
    duration_seconds: float | None = None
    speaker_count: int | None = None
    processing_seconds: float | None = None
    speed_ratio: float | None = None

    @classmethod
    def from_job(cls, job: Job) -> JobOut:
        return cls(
            id=job.id,
            filename=job.filename,
            status=job.status,
            stage=job.stage,
            progress=job.progress,
            created_at=job.created_at,
            error=job.error,
            duration_seconds=job.duration_seconds,
            speaker_count=job.speaker_count,
            processing_seconds=job.processing_seconds,
            speed_ratio=job.speed_ratio,
        )


class JobResultOut(JobOut):
    segments: list[SegmentOut] = []
    text: str = ""

    @classmethod
    def from_job_with_result(cls, job: Job, text: str) -> JobResultOut:
        base = JobOut.from_job(job).model_dump()
        return cls(
            **base,
            segments=[
                SegmentOut(
                    start=s.start, end=s.end, speaker=s.speaker, text=s.text
                )
                for s in job.segments
            ],
            text=text,
        )


class HealthOut(BaseModel):
    status: str
    engine: str
    device: str
    whisper_model: str
    diarization_model: str
    hf_token_configured: bool
    # Presente solo si se pidio GPU y hubo que degradar a CPU.
    fallback_reason: str | None = None
