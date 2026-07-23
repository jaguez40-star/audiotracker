import { Users } from 'lucide-react'

import styles from './SpeakerHint.module.scss'

// null = deteccion automatica. Mas de 6 personas es raro en una reunion y la
// lista se vuelve incomoda; para esos casos queda el modo automatico.
const OPCIONES: Array<{ valor: number | null; etiqueta: string }> = [
  { valor: null, etiqueta: 'Auto' },
  { valor: 1, etiqueta: '1' },
  { valor: 2, etiqueta: '2' },
  { valor: 3, etiqueta: '3' },
  { valor: 4, etiqueta: '4' },
  { valor: 5, etiqueta: '5' },
  { valor: 6, etiqueta: '6' },
]

interface Props {
  value: number | null
  onChange: (value: number | null) => void
  disabled?: boolean
}

export function SpeakerHint({ value, onChange, disabled = false }: Props) {
  return (
    <div className={styles.wrapper}>
      <span className={styles.label}>
        <Users size={16} className={styles.icon} />
        ¿Cuántas personas hablan?
      </span>

      <div className={styles.options} role="radiogroup" aria-label="Número de hablantes">
        {OPCIONES.map((op) => (
          <button
            key={op.etiqueta}
            className={styles.option}
            data-active={value === op.valor}
            role="radio"
            aria-checked={value === op.valor}
            disabled={disabled}
            onClick={() => onChange(op.valor)}
          >
            {op.etiqueta}
          </button>
        ))}
      </div>

      <span className={styles.hint}>
        {value === null
          ? 'En automático puede confundir una pausa larga con un cambio de persona. Si sabes el número, indícalo.'
          : `Se forzará exactamente ${value} ${value === 1 ? 'hablante' : 'hablantes'}.`}
      </span>
    </div>
  )
}
