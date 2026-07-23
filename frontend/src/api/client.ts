export type JobStatus = 'queued' | 'processing' | 'done' | 'error'

export type JobStage =
  | 'pending'
  | 'preparing'
  | 'transcribing'
  | 'diarizing'
  | 'merging'
  | 'finished'

export interface Job {
  id: string
  filename: string
  status: JobStatus
  stage: JobStage
  progress: number
  created_at: string
  error: string | null
  duration_seconds: number | null
  speaker_count: number | null
  processing_seconds: number | null
  speed_ratio: number | null
}

export interface Segment {
  start: number
  end: number
  speaker: string
  text: string
}

export interface JobResult extends Job {
  segments: Segment[]
  text: string
}

export interface Health {
  status: string
  engine: string
  device: string
  whisper_model: string
  diarization_model: string
  hf_token_configured: boolean
  fallback_reason: string | null
}

async function parseError(res: Response): Promise<string> {
  try {
    const body = await res.json()
    return typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
  } catch {
    return `${res.status} ${res.statusText}`
  }
}

async function get<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(await parseError(res))
  return res.json() as Promise<T>
}

export const api = {
  health: () => get<Health>('/api/health'),

  async upload(file: File, numSpeakers?: number | null): Promise<Job> {
    const form = new FormData()
    form.append('file', file)
    // Se omite el campo si no hay dato: el backend lo interpreta como
    // "detectar automaticamente".
    if (numSpeakers != null) form.append('num_speakers', String(numSpeakers))
    const res = await fetch('/api/transcribe', { method: 'POST', body: form })
    if (!res.ok) throw new Error(await parseError(res))
    return res.json() as Promise<Job>
  },

  job: (id: string) => get<Job>(`/api/jobs/${id}`),

  result: (id: string) => get<JobResult>(`/api/jobs/${id}/result`),

  downloadUrl: (id: string) => `/api/jobs/${id}/download`,
}

export const STAGE_LABELS: Record<JobStage, string> = {
  pending: 'En cola',
  preparing: 'Preparando audio',
  transcribing: 'Transcribiendo',
  diarizing: 'Identificando hablantes',
  merging: 'Armando transcript',
  finished: 'Listo',
}

export function formatDuration(seconds: number): string {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  const pad = (n: number) => String(n).padStart(2, '0')
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`
}
