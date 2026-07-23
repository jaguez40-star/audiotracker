import { Check, Copy, Download, RotateCcw } from 'lucide-react'
import { useMemo, useState } from 'react'

import { api, formatDuration, type JobResult } from '../api/client'
import styles from './TranscriptView.module.scss'

const SPEAKER_VARS = [
  'var(--speaker-1)',
  'var(--speaker-2)',
  'var(--speaker-3)',
  'var(--speaker-4)',
  'var(--speaker-5)',
  'var(--speaker-6)',
]

function timestamp(seconds: number): string {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${pad(h)}:${pad(m)}:${pad(s)}`
}

interface Props {
  result: JobResult
  onReset: () => void
}

export function TranscriptView({ result, onReset }: Props) {
  const [copied, setCopied] = useState(false)

  // Color estable por hablante: se asigna por orden de aparicion, no por el
  // indice del segmento, para que un mismo hablante conserve su color.
  const colorOf = useMemo(() => {
    const mapa = new Map<string, string>()
    for (const seg of result.segments) {
      if (!mapa.has(seg.speaker)) {
        mapa.set(seg.speaker, SPEAKER_VARS[mapa.size % SPEAKER_VARS.length])
      }
    }
    return mapa
  }, [result.segments])

  async function copy() {
    await navigator.clipboard.writeText(result.text)
    setCopied(true)
    setTimeout(() => setCopied(false), 1800)
  }

  const speakers = result.speaker_count ?? colorOf.size

  return (
    <div className={styles.wrapper}>
      <div className={styles.toolbar}>
        <div className={styles.stats}>
          <span className={styles.filename}>{result.filename}</span>
          <span className={styles.summary}>
            {speakers} {speakers === 1 ? 'hablante' : 'hablantes'} ·{' '}
            {result.segments.length} intervenciones
            {result.duration_seconds !== null &&
              ` · ${formatDuration(result.duration_seconds)}`}
            {result.processing_seconds !== null && (
              <span className={styles.timing}>
                {' · '}procesado en {formatDuration(result.processing_seconds)}
                {result.speed_ratio !== null &&
                  ` (${result.speed_ratio.toFixed(1)}× la duración)`}
              </span>
            )}
          </span>
        </div>

        <button className={styles.action} onClick={copy}>
          {copied ? <Check size={15} /> : <Copy size={15} />}
          {copied ? 'Copiado' : 'Copiar'}
        </button>

        <button className={styles.action} onClick={onReset}>
          <RotateCcw size={15} />
          Otro archivo
        </button>

        <a
          className={`${styles.action} ${styles.primaryAction}`}
          href={api.downloadUrl(result.id)}
          download
        >
          <Download size={15} />
          Descargar .txt
        </a>
      </div>

      <div className={styles.transcript}>
        {result.segments.map((seg, i) => (
          <div
            key={`${seg.start}-${i}`}
            className={styles.turn}
            style={
              {
                '--speaker-color': colorOf.get(seg.speaker),
              } as React.CSSProperties
            }
          >
            <div className={styles.gutter}>
              <span className={styles.speaker}>{seg.speaker}</span>
              <span className={styles.time}>{timestamp(seg.start)}</span>
            </div>
            <p className={styles.text}>{seg.text}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
