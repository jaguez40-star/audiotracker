"""Orquestacion del procesamiento de un archivo de audio.

El orden importa por memoria, no solo por logica: los modelos se cargan y se
liberan de a uno. Con 8 GB de RAM, mantener Whisper y pyannote vivos al mismo
tiempo lleva la maquina a swap y el proceso se vuelve mas lento que el propio
computo. Los bloques `with` garantizan que solo haya un modelo en RAM a la vez.

Con whisper.cpp la transcripcion ni siquiera ocupa RAM del backend: el modelo
vive en la VRAM del subproceso. La diarizacion sigue en CPU en cualquier caso,
porque pyannote es PyTorch y no hay build con GPU para esta tarjeta.
"""

from __future__ import annotations

import traceback
from datetime import datetime, timezone
from pathlib import Path

from app.audio import probe_duration, to_wav_16k_mono
from app.core.config import settings
from app.core.logging import get_logger
from app.engines import build_transcription_engine
from app.engines.pyannote_engine import PyannoteEngine
from app.jobs import JobStage, JobStatus, store
from app.merge import build_transcript, to_plain_text

log = get_logger(__name__)


def process_audio(
    job_id: str, source_path: Path, num_speakers: int | None = None
) -> None:
    """Ejecuta el pipeline completo. Pensado para correr en background.

    num_speakers viene de la peticion: si quien sube el audio sabe cuanta gente
    habla, esa pista mejora la diarizacion mas que cualquier ajuste de modelo.
    """
    wav_path = settings.upload_dir / f"{job_id}.wav"
    log_ctx = log.bind(job_id=job_id, archivo=source_path.name)

    try:
        store.update(
            job_id,
            status=JobStatus.PROCESSING,
            started_at=datetime.now(timezone.utc),
        )

        # --- 1. Normalizar audio ---
        store.set_stage_progress(job_id, JobStage.PREPARING, 0.0)
        to_wav_16k_mono(source_path, wav_path)
        duracion = probe_duration(wav_path)
        store.update(job_id, duration_seconds=duracion)
        store.set_stage_progress(job_id, JobStage.PREPARING, 1.0)

        # --- 2. Transcribir (modelo cargado y liberado aqui dentro) ---
        # La etapa se marca antes de instanciar el motor para que un fallo al
        # cargar el modelo quede atribuido a la etapa correcta y no a la previa.
        store.set_stage_progress(job_id, JobStage.TRANSCRIBING, 0.0)
        with build_transcription_engine() as engine:
            segmentos = engine.transcribe(
                str(wav_path),
                on_progress=lambda f: store.set_stage_progress(
                    job_id, JobStage.TRANSCRIBING, f
                ),
            )

        # --- 3. Diarizar (idem: el modelo anterior ya se libero) ---
        store.set_stage_progress(job_id, JobStage.DIARIZING, 0.0)
        with PyannoteEngine() as engine:
            turnos = engine.diarize(
                str(wav_path),
                on_progress=lambda f: store.set_stage_progress(
                    job_id, JobStage.DIARIZING, f
                ),
                num_speakers=num_speakers,
            )

        # --- 4. Cruzar y serializar ---
        store.set_stage_progress(job_id, JobStage.MERGING, 0.0)
        transcript = build_transcript(segmentos, turnos)
        texto = to_plain_text(transcript)

        text_path = settings.output_dir / f"{job_id}.txt"
        text_path.write_text(texto, encoding="utf-8")

        hablantes = len({s.speaker for s in transcript})
        store.update(
            job_id,
            status=JobStatus.DONE,
            stage=JobStage.FINISHED,
            progress=1.0,
            segments=transcript,
            text_path=text_path,
            speaker_count=hablantes,
            finished_at=datetime.now(timezone.utc),
        )
        job = store.get(job_id)
        log_ctx.info(
            "procesamiento completo",
            bloques=len(transcript),
            hablantes=hablantes,
            computo_s=round(job.processing_seconds or 0, 1),
            ratio=round(job.speed_ratio or 0, 2),
        )

    except Exception as exc:
        log_ctx.error("procesamiento fallido", error=str(exc), trace=traceback.format_exc())
        store.update(
            job_id,
            status=JobStatus.ERROR,
            error=str(exc),
            finished_at=datetime.now(timezone.utc),
        )

    finally:
        _cleanup(wav_path, source_path, log_ctx)


def _cleanup(wav_path: Path, source_path: Path, log_ctx) -> None:
    """El WAV normalizado siempre sobra; el original depende de configuracion."""
    wav_path.unlink(missing_ok=True)
    if not settings.keep_uploads:
        source_path.unlink(missing_ok=True)
    log_ctx.debug("temporales limpiados")
