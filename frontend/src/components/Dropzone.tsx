import { AlertCircle, UploadCloud } from 'lucide-react'
import { useRef, useState } from 'react'

import styles from './Dropzone.module.scss'

const ACCEPTED = '.mp4,.wav,.mp3,.m4a,.aac,.flac,.ogg,.opus,.webm,.mkv,.mov'

interface Props {
  onFile: (file: File) => void
  disabled?: boolean
  error?: string | null
}

export function Dropzone({ onFile, disabled = false, error }: Props) {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  // El navegador dispara dragenter/dragleave por cada elemento hijo que se
  // cruza, asi que un booleano parpadearia. Contamos entradas y salidas.
  const dragDepth = useRef(0)

  function handleDrop(event: React.DragEvent) {
    event.preventDefault()
    dragDepth.current = 0
    setDragging(false)
    const file = event.dataTransfer.files?.[0]
    if (file) onFile(file)
  }

  return (
    <div>
      <div
        className={styles.zone}
        data-dragging={dragging}
        data-disabled={disabled}
        role="button"
        tabIndex={disabled ? -1 : 0}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault()
            inputRef.current?.click()
          }
        }}
        onDragEnter={(e) => {
          e.preventDefault()
          dragDepth.current += 1
          setDragging(true)
        }}
        onDragOver={(e) => e.preventDefault()}
        onDragLeave={(e) => {
          e.preventDefault()
          dragDepth.current -= 1
          if (dragDepth.current <= 0) setDragging(false)
        }}
        onDrop={handleDrop}
      >
        <UploadCloud size={38} strokeWidth={1.5} className={styles.icon} />
        <span className={styles.primary}>Arrastra un audio o haz clic para elegirlo</span>
        <span className={styles.secondary}>
          Se transcribe y se separa por hablante. Todo el procesamiento ocurre en
          esta máquina — nada sale a internet.
        </span>
        <span className={styles.formats}>mp4 · wav · mp3 · m4a · flac · ogg · webm</span>

        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED}
          className={styles.input}
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (file) onFile(file)
            // Permite volver a elegir el mismo archivo tras un error.
            e.target.value = ''
          }}
        />
      </div>

      {error && (
        <div className={styles.error}>
          <AlertCircle size={16} />
          <span>{error}</span>
        </div>
      )}
    </div>
  )
}
