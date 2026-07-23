"""Pruebas del cruce transcripcion/diarizacion.

Solo tocan funciones puras: no cargan modelos ni requieren torch.
"""

from __future__ import annotations

from app.engines.base import SpeakerTurn, TranscriptionSegment
from app.merge import (
    assign_speakers,
    build_transcript,
    format_timestamp,
    merge_consecutive,
    relabel_by_appearance,
    to_plain_text,
)


def seg(start: float, end: float, text: str) -> TranscriptionSegment:
    return TranscriptionSegment(start=start, end=end, text=text)


def turn(start: float, end: float, speaker: str) -> SpeakerTurn:
    return SpeakerTurn(start=start, end=end, speaker=speaker)


class TestAssignSpeakers:
    def test_asigna_por_solape_simple(self):
        segmentos = [seg(0, 5, "hola"), seg(6, 10, "que tal")]
        turnos = [turn(0, 5, "SPEAKER_00"), turn(6, 10, "SPEAKER_01")]

        resultado = assign_speakers(segmentos, turnos)

        assert [s.speaker for s in resultado] == ["SPEAKER_00", "SPEAKER_01"]

    def test_gana_el_hablante_con_mayor_solape_acumulado(self):
        # El segmento cruza un cambio de hablante: 1s del primero, 4s del segundo.
        # Debe quedar con el segundo, aunque el primero aparezca antes.
        segmentos = [seg(4, 9, "texto a caballo")]
        turnos = [turn(0, 5, "SPEAKER_00"), turn(5, 12, "SPEAKER_01")]

        resultado = assign_speakers(segmentos, turnos)

        assert resultado[0].speaker == "SPEAKER_01"

    def test_suma_solapes_no_consecutivos_del_mismo_hablante(self):
        # SPEAKER_00 aparece en dos tramos cortos que suman mas que el tramo
        # unico de SPEAKER_01. Tomar solo el maximo individual daria SPEAKER_01.
        segmentos = [seg(0, 10, "largo")]
        turnos = [
            turn(0, 3, "SPEAKER_00"),
            turn(3, 7, "SPEAKER_01"),
            turn(7, 10, "SPEAKER_00"),
        ]

        resultado = assign_speakers(segmentos, turnos)

        assert resultado[0].speaker == "SPEAKER_00"

    def test_sin_solape_cae_al_turno_mas_cercano(self):
        segmentos = [seg(20, 22, "texto huerfano")]
        turnos = [turn(0, 5, "SPEAKER_00"), turn(24, 30, "SPEAKER_01")]

        resultado = assign_speakers(segmentos, turnos)

        assert resultado[0].speaker == "SPEAKER_01"

    def test_sin_turnos_no_pierde_texto(self):
        segmentos = [seg(0, 5, "hola"), seg(5, 8, "adios")]

        resultado = assign_speakers(segmentos, [])

        assert len(resultado) == 2
        assert [s.text for s in resultado] == ["hola", "adios"]


class TestRelabel:
    def test_renumera_por_orden_de_aparicion(self):
        # pyannote entrega SPEAKER_01 primero; debe volverse "Speaker 1".
        segmentos = assign_speakers(
            [seg(0, 5, "primero"), seg(5, 10, "segundo")],
            [turn(0, 5, "SPEAKER_07"), turn(5, 10, "SPEAKER_02")],
        )

        resultado = relabel_by_appearance(segmentos)

        assert resultado[0].speaker == "Speaker 1"
        assert resultado[1].speaker == "Speaker 2"

    def test_mismo_hablante_conserva_etiqueta(self):
        segmentos = assign_speakers(
            [seg(0, 5, "a"), seg(5, 10, "b"), seg(10, 15, "c")],
            [turn(0, 5, "SPEAKER_03"), turn(5, 10, "SPEAKER_01"), turn(10, 15, "SPEAKER_03")],
        )

        resultado = relabel_by_appearance(segmentos)

        assert [s.speaker for s in resultado] == ["Speaker 1", "Speaker 2", "Speaker 1"]


class TestMergeConsecutive:
    def test_une_bloques_contiguos_del_mismo_hablante(self):
        transcript = build_transcript(
            [seg(0, 3, "Hola."), seg(3, 6, "Como estas?"), seg(6, 9, "Bien.")],
            [turn(0, 6, "SPEAKER_00"), turn(6, 9, "SPEAKER_01")],
        )

        assert len(transcript) == 2
        assert transcript[0].text == "Hola. Como estas?"
        assert transcript[0].start == 0
        assert transcript[0].end == 6
        assert transcript[1].text == "Bien."

    def test_lista_vacia(self):
        assert merge_consecutive([]) == []

    def test_alternancia_no_se_une(self):
        transcript = build_transcript(
            [seg(0, 2, "a"), seg(2, 4, "b"), seg(4, 6, "c")],
            [turn(0, 2, "S0"), turn(2, 4, "S1"), turn(4, 6, "S0")],
        )

        assert len(transcript) == 3


class TestFormato:
    def test_timestamp(self):
        assert format_timestamp(0) == "00:00:00"
        assert format_timestamp(65) == "00:01:05"
        assert format_timestamp(3725) == "01:02:05"

    def test_texto_plano_con_timestamps(self):
        transcript = build_transcript(
            [seg(0, 5, "Hola."), seg(65, 70, "Que tal.")],
            [turn(0, 5, "S0"), turn(65, 70, "S1")],
        )

        salida = to_plain_text(transcript)

        assert "[00:00:00] Speaker 1: Hola." in salida
        assert "[00:01:05] Speaker 2: Que tal." in salida

    def test_texto_plano_sin_timestamps(self):
        transcript = build_transcript([seg(0, 5, "Hola.")], [turn(0, 5, "S0")])

        salida = to_plain_text(transcript, with_timestamps=False)

        assert salida == "Speaker 1: Hola."
