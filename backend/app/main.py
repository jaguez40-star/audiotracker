"""Punto de entrada de la API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.routes import router

configure_logging()
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    log.info(
        "audio_track iniciado",
        whisper=settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        hf_token=bool(settings.hf_token),
    )
    if not settings.hf_token:
        log.warning("HF_TOKEN no configurado: la diarizacion fallara. Ver README.")
    yield


app = FastAPI(
    title="audio_track",
    description="Transcripcion de audio con identificacion de hablantes",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
