"""Cruce de transcripcion y diarizacion.

Whisper produce segmentos de texto con tiempos; pyannote produce intervalos de
habla por persona. Ninguno de los dos sabe del otro, y sus fronteras no
coinciden: un segmento de texto puede solapar dos turnos distintos. Aqui se
resuelve a quien pertenece cada frase.

Funciones puras, sin dependencias de los motores: se pueden probar con datos
sinteticos.
"""

from __future__ import annotations

from app.engines.base import SpeakerSegment, SpeakerTurn, TranscriptionSegment


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Segundos de solape entre dos intervalos. Cero si no se tocan."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speakers(
    segments: list[TranscriptionSegment],
    turns: list[SpeakerTurn],
) -> list[SpeakerSegment]:
    """Atribuye cada segmento de texto al hablante con mayor solape temporal.

    Se acumula el solape por hablante en vez de tomar el primer turno que toca,
    porque un segmento que cruza un cambio de hablante debe quedar con quien
    hablo mas rato dentro de el, no con quien aparecio primero.
    """
    if not turns:
        return [
            SpeakerSegment(s.start, s.end, "SPEAKER_00", s.text) for s in segments
        ]

    resultado: list[SpeakerSegment] = []
    for seg in segments:
        por_hablante: dict[str, float] = {}
        for turn in turns:
            solape = _overlap(seg.start, seg.end, turn.start, turn.end)
            if solape > 0:
                por_hablante[turn.speaker] = por_hablante.get(turn.speaker, 0.0) + solape

        if por_hablante:
            hablante = max(por_hablante, key=lambda k: por_hablante[k])
        else:
            # Sin solape: pyannote no detecto voz aqui (musica, ruido, silencio
            # mal recortado). Lo asignamos al turno mas cercano en el tiempo, que
            # es mejor aproximacion que descartar el texto.
            medio = (seg.start + seg.end) / 2
            hablante = min(
                turns,
                key=lambda t: min(abs(t.start - medio), abs(t.end - medio)),
            ).speaker

        resultado.append(SpeakerSegment(seg.start, seg.end, hablante, seg.text))

    return resultado


def relabel_by_appearance(segments: list[SpeakerSegment]) -> list[SpeakerSegment]:
    """Renombra las etiquetas de pyannote a 'Speaker 1', 'Speaker 2'...

    pyannote entrega SPEAKER_00, SPEAKER_01... en un orden que no corresponde al
    de aparicion. Renumerar por primera aparicion hace que 'Speaker 1' sea
    siempre quien habla primero, que es lo que espera quien lee el resultado.
    """
    mapa: dict[str, str] = {}
    for seg in segments:
        if seg.speaker not in mapa:
            mapa[seg.speaker] = f"Speaker {len(mapa) + 1}"

    return [
        SpeakerSegment(s.start, s.end, mapa[s.speaker], s.text) for s in segments
    ]


def merge_consecutive(segments: list[SpeakerSegment]) -> list[SpeakerSegment]:
    """Une segmentos contiguos del mismo hablante en un solo bloque.

    Whisper corta por pausas prosodicas, asi que una intervencion continua sale
    partida en varias frases. Unirlas produce un transcript legible en lugar de
    una lista de fragmentos con el mismo nombre repetido.
    """
    if not segments:
        return []

    bloques: list[SpeakerSegment] = []
    actual = segments[0]

    for seg in segments[1:]:
        if seg.speaker == actual.speaker:
            actual = SpeakerSegment(
                start=actual.start,
                end=seg.end,
                speaker=actual.speaker,
                text=f"{actual.text} {seg.text}".strip(),
            )
        else:
            bloques.append(actual)
            actual = seg

    bloques.append(actual)
    return bloques


def build_transcript(
    segments: list[TranscriptionSegment],
    turns: list[SpeakerTurn],
) -> list[SpeakerSegment]:
    """Pipeline completo de cruce: atribuir -> renombrar -> unir."""
    return merge_consecutive(relabel_by_appearance(assign_speakers(segments, turns)))


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def to_plain_text(segments: list[SpeakerSegment], with_timestamps: bool = True) -> str:
    """Serializa a texto plano, listo para leer o para pasarle a un LLM."""
    lineas = []
    for seg in segments:
        prefijo = f"[{format_timestamp(seg.start)}] " if with_timestamps else ""
        lineas.append(f"{prefijo}{seg.speaker}: {seg.text}")
    return "\n\n".join(lineas)
