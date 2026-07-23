import { useMutation, useQuery } from '@tanstack/react-query'
import { AlertTriangle, AudioLines } from 'lucide-react'
import { useState } from 'react'

import { api, type Job } from './api/client'
import styles from './App.module.scss'
import { Dropzone } from './components/Dropzone'
import { JobProgress } from './components/JobProgress'
import { SpeakerHint } from './components/SpeakerHint'
import { TranscriptView } from './components/TranscriptView'

// La subida ocurre antes de que exista un job en el backend, pero merece la
// misma tarjeta de progreso para que la transicion no parpadee.
const UPLOADING_PLACEHOLDER: Job = {
  id: 'uploading',
  filename: 'Subiendo archivo…',
  status: 'queued',
  stage: 'pending',
  progress: 0,
  created_at: '',
  error: null,
  duration_seconds: null,
  speaker_count: null,
  processing_seconds: null,
  speed_ratio: null,
}

export default function App() {
  const [jobId, setJobId] = useState<string | null>(null)
  const [numSpeakers, setNumSpeakers] = useState<number | null>(null)

  const health = useQuery({
    queryKey: ['health'],
    queryFn: api.health,
    // Se refresca sola: si se cambia de motor o se reinicia el backend, la
    // insignia debe reflejarlo sin obligar a recargar la pagina. Antes se
    // quedaba mostrando la configuracion vieja indefinidamente.
    staleTime: 15_000,
    refetchInterval: 30_000,
    refetchOnWindowFocus: true,
  })

  const job = useQuery({
    queryKey: ['job', jobId],
    queryFn: () => api.job(jobId!),
    enabled: jobId !== null,
    // Solo se consulta mientras hay trabajo en curso; al terminar se detiene y
    // el resultado se pide una unica vez.
    refetchInterval: (query) => {
      const status = query.state.data?.status
      return status === 'queued' || status === 'processing' ? 2000 : false
    },
  })

  const result = useQuery({
    queryKey: ['result', jobId],
    queryFn: () => api.result(jobId!),
    enabled: jobId !== null && job.data?.status === 'done',
    staleTime: Infinity,
  })

  const upload = useMutation({
    mutationFn: (file: File) => api.upload(file, numSpeakers),
    onSuccess: (created) => setJobId(created.id),
  })

  function reset() {
    setJobId(null)
    upload.reset()
  }

  const tokenMissing = health.data && !health.data.hf_token_configured

  return (
    <div className={styles.shell}>
      <div className={styles.container}>
        <header className={styles.header}>
          <div className={styles.logo}>
            <AudioLines size={22} />
          </div>
          <div className={styles.titles}>
            <span className={styles.title}>audio_track</span>
            <span className={styles.subtitle}>
              Transcripción local con identificación de hablantes
            </span>
          </div>
          {health.data && (
            <div
              className={styles.badge}
              title={`${health.data.engine} · ${health.data.whisper_model}`}
            >
              <span className={styles.dot} data-warn={!health.data.hf_token_configured} />
              {health.data.whisper_model} · {health.data.device}
            </div>
          )}
        </header>

        {tokenMissing && (
          <div className={styles.warning}>
            <AlertTriangle size={17} className={styles.warnIcon} />
            <div>
              Falta configurar <code>HF_TOKEN</code> en{' '}
              <code>backend/.env</code>. Sin él la diarización no puede correr y
              las subidas serán rechazadas. Los pasos están en el README.
            </div>
          </div>
        )}

        {/* Estados mutuamente excluyentes: subiendo -> procesando -> resultado,
            y el dropzone solo cuando no hay nada en curso. */}
        {upload.isPending ? (
          <JobProgress job={UPLOADING_PLACEHOLDER} onReset={reset} />
        ) : jobId === null ? (
          <>
            <SpeakerHint
              value={numSpeakers}
              onChange={setNumSpeakers}
              disabled={Boolean(tokenMissing)}
            />
            <Dropzone
              onFile={(file) => upload.mutate(file)}
              disabled={Boolean(tokenMissing)}
              error={upload.error?.message ?? null}
            />
          </>
        ) : result.data ? (
          <TranscriptView result={result.data} onReset={reset} />
        ) : job.data ? (
          <JobProgress job={job.data} onReset={reset} />
        ) : null}
      </div>
    </div>
  )
}
