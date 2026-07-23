import { AlertTriangle, Clock, Loader2 } from 'lucide-react'

import { formatDuration, STAGE_LABELS, type Job } from '../api/client'
import styles from './JobProgress.module.scss'

interface Props {
  job: Job
  onReset: () => void
}

export function JobProgress({ job, onReset }: Props) {
  if (job.status === 'error') {
    return (
      <div className={styles.failed}>
        <div className={styles.failedTop}>
          <AlertTriangle size={18} />
          <span>No se pudo procesar «{job.filename}»</span>
        </div>
        <div className={styles.failedMessage}>{job.error ?? 'Error desconocido'}</div>
        <button className={styles.retry} onClick={onReset}>
          Intentar con otro archivo
        </button>
      </div>
    )
  }

  const percent = Math.round(job.progress * 100)

  return (
    <div className={styles.card}>
      <div className={styles.top}>
        <Loader2 size={20} className={styles.spinner} />
        <div className={styles.meta}>
          <span className={styles.filename}>{job.filename}</span>
          <span className={styles.stage}>
            {STAGE_LABELS[job.stage]}
            {job.duration_seconds !== null &&
              ` · ${formatDuration(job.duration_seconds)} de audio`}
          </span>
        </div>
        <span className={styles.percent}>{percent}%</span>
      </div>

      <div
        className={styles.track}
        role="progressbar"
        aria-valuenow={percent}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label="Progreso del procesamiento"
      >
        <div className={styles.fill} style={{ width: `${percent}%` }} />
      </div>

      <div className={styles.note}>
        <Clock size={14} className={styles.noteIcon} />
        <span>
          Esto corre en CPU y toma su tiempo. La identificación de hablantes es la
          etapa más lenta — bastante más que la transcripción. Puedes dejar la
          pestaña abierta y volver después.
        </span>
      </div>
    </div>
  )
}
