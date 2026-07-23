# CLAUDE.md — audio_track: Transcripción local con identificación de hablantes

Herramienta local de un solo usuario. Recibe un archivo de audio, lo transcribe y devuelve el texto separado por hablante (`Speaker 1`, `Speaker 2`…) en un `.txt`. Sin login, sin base de datos, página única.

**Fase 1** (actual): transcripción + diarización + descarga `.txt`.
**Fase 2** (siguiente): entregar ese `.txt` a un LLM local vía Ollama para procesarlo.

---

## 1. Idioma y estilo de comunicación

- **Todo en español** (mensajes, respuestas, comentarios de código, commits, ramas).
- **Respuestas breves y directas.** Sin preámbulos ("Claro", "Por supuesto"). No explicar antes de hacer ni resumir después salvo que el resultado lo amerite. Si la respuesta es código, entregar el código. Máximo 2-3 líneas de explicación cuando haga falta. Tuteo informal.

---

## 2. Stack tecnológico

### Backend (`backend/`)

| Componente | Versión | Uso |
|------------|---------|-----|
| Python | 3.12.10 | Lenguaje base |
| FastAPI | 0.139 | API REST + OpenAPI automático |
| Pydantic | 2.13 | Validación |
| pydantic-settings | 2.14 | Configuración desde `.env` |
| structlog | 26.x | Logs JSON UTC |
| uvicorn | 0.51 | Servidor ASGI |
| pytest | 8.x | Tests |
| pip + venv | — | Gestor de paquetes (`backend/venv/`) |

### Motores de audio

| Componente | Versión | Dispositivo | Uso |
|------------|---------|-------------|-----|
| **whisper.cpp** | v1.8.4 (lemonade-sdk) | **GPU (Vulkan)** | Transcripción — motor por defecto |
| faster-whisper | 1.2 | CPU | Transcripción — respaldo automático |
| pyannote.audio | 4.0.7 | CPU | Diarización — sin alternativa GPU |
| PyAV (`av`) | 18.0 | CPU | Decodificación de audio |
| soundfile | 0.14 | CPU | Lectura WAV para pyannote |

### Frontend (`frontend/`)

| Componente | Versión | Uso |
|------------|---------|-----|
| React | 19 | UI |
| TypeScript | 5.7 | Tipado estricto |
| Vite | 6 | Build y dev server |
| TanStack Query | 5 | Estado de servidor + polling de jobs |
| Sass Modules | 1.83 | Estilos |
| Lucide React | 0.468 | Iconografía |
| npm | — | Gestor de paquetes |

### Puertos

| Componente | URL | Arranque |
|-----------|-----|----------|
| Frontend (Vite) | `http://localhost:6023` | `Start_Front.bat` |
| Backend (uvicorn) | `http://127.0.0.1:6024` | `Start_Back.bat` |
| Docs API | `http://127.0.0.1:6024/docs` | — |

**Sin base de datos.** Los jobs viven en memoria (`app/jobs.py`) y se pierden al reiniciar; los transcripts quedan en `backend/data/outputs/`.

---

## 3. Arquitectura

### Estructura de directorios

Estado real verificado al 2026-07-23. Lo marcado 🚫 no se versiona (ver `.gitignore`).

```
audio_track/
├── CLAUDE.md                       # este archivo
├── README.md                       # documentación de usuario
├── .gitignore
├── Start_Back.bat                  # uvicorn :6024 (valida venv y .env)
├── Start_Front.bat                 # vite :6023 (npm install si falta)
├── vendor/whisper-vulkan/       🚫 # whisper-cli.exe + ggml-vulkan.dll (54 MB)
├── models/                      🚫 # ggml-large-v3-q5_0.bin (1,01 GB)
│
├── backend/
│   ├── .env                     🚫 # HF_TOKEN + configuración activa
│   ├── .env.example                # plantilla documentada
│   ├── requirements.txt            # producción
│   ├── requirements-dev.txt        # + pytest, httpx
│   ├── pytest.ini                  # pythonpath=. · testpaths=tests
│   ├── venv/                    🚫
│   ├── data/                    🚫 # uploads/ (efímero) + outputs/ (.txt)
│   ├── scripts/
│   │   ├── check_setup.py          # preflight: token, licencias, motor, códecs
│   │   └── benchmark_engines.py    # comparativa GPU vs CPU vs faster-whisper
│   ├── tests/                      # 36 tests, sin cargar modelos
│   │   ├── test_merge.py           # atribución de hablantes (19)
│   │   ├── test_api.py             # contratos de endpoints (11)
│   │   └── test_whispercpp.py      # comando y parseo JSON (14)
│   └── app/
│       ├── main.py                 # FastAPI + lifespan + CORS
│       ├── routes.py               # 6 endpoints
│       ├── schemas.py              # JobOut · JobResultOut · HealthOut
│       ├── jobs.py                 # store en memoria + pesos de etapa + timing
│       ├── pipeline.py             # orquestación con carga secuencial
│       ├── audio.py                # normalización a WAV 16 kHz mono (PyAV)
│       ├── merge.py                # cruce transcripción ↔ hablantes (puro)
│       ├── core/
│       │   ├── config.py           # Settings + _blank_to_none
│       │   └── logging.py          # structlog JSON + silencio de avisos
│       └── engines/
│           ├── base.py                  # protocolos + dataclasses del dominio
│           ├── __init__.py              # selector + fallback + describe()
│           ├── whispercpp_engine.py     # GPU Vulkan vía subproceso
│           ├── faster_whisper_engine.py # CPU (respaldo)
│           └── pyannote_engine.py       # diarización CPU
│
└── frontend/
    ├── index.html
    ├── package.json  ·  vite.config.ts  ·  tsconfig.json
    ├── node_modules/            🚫
    └── src/
        ├── main.tsx                # QueryClient + StrictMode
        ├── App.tsx                 # estados excluyentes + selector de hablantes
        ├── App.module.scss
        ├── vite-env.d.ts
        ├── api/client.ts           # tipos + cliente HTTP + helpers de formato
        ├── components/
        │   ├── Dropzone.tsx        # drag&drop con conteo de profundidad
        │   ├── SpeakerHint.tsx     # Auto · 1-6 hablantes
        │   ├── JobProgress.tsx     # barra + etapa + estado de error
        │   └── TranscriptView.tsx  # transcript, copiar, descargar
        └── styles/global.scss      # tokens claro/oscuro + paleta de hablantes
```

### Reglas de arquitectura

- **Motores detrás de protocolos.** `TranscriptionEngine` y `DiarizationEngine` en `engines/base.py` son el contrato. Cambiar de motor = escribir uno nuevo que lo cumpla; el pipeline, la API y el frontend no se tocan. Esta decisión ya se pagó sola: añadir la GPU no requirió reescribir nada.
- **Carga secuencial de modelos.** Solo un modelo en memoria a la vez, garantizado por bloques `with`. Con 8 GB de RAM, dos modelos vivos llevan la máquina a swap.
- **`merge.py` son funciones puras.** Sin dependencias de motores ni de I/O, testeables con datos sintéticos.
- **El contrato JSON de la API es `snake_case`.** El frontend consume esos nombres tal cual — no hay capa de mappers porque no hay volumen que lo justifique.

### Pipeline

```
audio → normalizar a WAV 16k mono (PyAV)
      → transcribir      (whisper.cpp GPU · libera)
      → diarizar         (pyannote CPU  · libera)
      → cruzar y escribir .txt
```

---

## 3-bis. Funcionalidades incorporadas

Estado operativo al 2026-07-23. Todo lo listado está implementado y verificado.

### API — 6 endpoints

| Método | Ruta | Función |
|---|---|---|
| `GET` | `/api/health` | Motor activo, dispositivo, modelo, token, motivo de fallback |
| `POST` | `/api/transcribe` | Sube audio + `num_speakers` opcional → job en background (202) |
| `GET` | `/api/jobs` | Lista de jobs, más reciente primero |
| `GET` | `/api/jobs/{id}` | Estado, etapa, progreso, tiempos |
| `GET` | `/api/jobs/{id}/result` | Transcript completo en JSON |
| `GET` | `/api/jobs/{id}/download` | Descarga `.txt` |

### Procesamiento

- **11 formatos de entrada**: `mp4 · wav · mp3 · m4a · aac · flac · ogg · opus · webm · mkv · mov`. Los contenedores de vídeo se aceptan y se extrae solo el audio.
- **Normalización previa** a WAV 16 kHz mono, para que ambos motores reciban lo mismo y no se decodifique dos veces.
- **Transcripción en GPU** (whisper.cpp + Vulkan, `large-v3`) con **degradación automática** a faster-whisper CPU si falta binario o modelo.
- **Diarización** con pyannote 4.x en CPU, con pista opcional del número de hablantes.
- **Carga secuencial de modelos** — nunca dos en memoria a la vez.
- **Atribución por solape acumulado**: cada frase va al hablante con más segundos de solape, no al primer turno que toca. Renumerado por orden de aparición, de modo que `Speaker 1` siempre es quien habla primero.
- **Unión de bloques contiguos** del mismo hablante, para un transcript legible.
- **Progreso ponderado por etapa** (preparando 3% · transcribiendo 25% · diarizando 69% · cruzando 3%), monótono: nunca retrocede.
- **Medición de rendimiento**: `processing_seconds` y `speed_ratio` por job.
- **Limpieza automática** de temporales; el original se conserva solo si `KEEP_UPLOADS=true`.

### Interfaz

- Página única con dropzone (clic o arrastrar) y validación de formato.
- **Selector de hablantes** `Auto · 1 · 2 · 3 · 4 · 5 · 6`, por archivo.
- Progreso en vivo con etapa nombrada, mediante polling cada 2 s.
- Transcript con **color estable por hablante** y marca de tiempo por intervención.
- Copiar al portapapeles · descargar `.txt` · procesar otro archivo.
- Aviso destacado si falta `HF_TOKEN`, con el dropzone deshabilitado.
- Insignia de motor activo, auto-refrescada cada 30 s.
- Tema claro/oscuro automático · responsive · `prefers-reduced-motion`.

### Herramientas de operación

| Script | Función |
|---|---|
| `scripts/check_setup.py` | Preflight: token, **descarga real** de los 3 repos con licencia, motor activo, códecs |
| `scripts/benchmark_engines.py` | Compara GPU vs CPU vs faster-whisper sobre el mismo audio |

### Aún no implementado

Salida en `.docx`/Excel · procesamiento por lotes · persistencia de jobs entre reinicios · identificación nominal de hablantes (solo `Speaker N`) · Fase 2 con Ollama.

---

## 4. Entorno de ejecución — restricciones verificadas

Estas restricciones se comprobaron empíricamente, no se asumieron. **No re-litigarlas sin datos nuevos.**

| Recurso | Estado |
|---|---|
| GPU | AMD RX 6700 XT · RDNA2 · **gfx1031** · 12 GB VRAM |
| CPU | Intel i5-10400 · 6 núcleos / 12 hilos |
| RAM | **7,9 GB** — es la restricción que manda el diseño |
| Smart App Control | **ACTIVO** — bloquea binarios sin firma ni reputación |

### Rutas de GPU descartadas

| Ruta | Resultado | Evidencia |
|---|---|---|
| CUDA | ❌ | No hay GPU NVIDIA |
| PyTorch + ROCm Windows | ❌ | Solo gfx1100/1101/1200/1201 (RDNA3/4) |
| CTranslate2 + ROCm | ❌ | Compila gfx1030 y gfx1100+, no gfx1031 |
| `HSA_OVERRIDE_GFX_VERSION` | ❌ | Exclusivo de Linux |
| torch-directml | ❌ | Abandonado (sept 2024, torch 2.4.1) |
| **Vulkan** | ✅ | Ollama: 100% GPU, **51 vs 9,5 tok/s** |

Ollama sirve de banco de pruebas porque usa la misma librería `ggml` que whisper.cpp. Su detector rechaza ROCm con `hipGetDeviceCount failed: 100`. Los BSODs con Vulkan que documenta ProdIA eran de un driver anterior; el 26.6.4 los resolvió.

### Consecuencias en el código

- **Binario precompilado, nunca compilar.** Smart App Control bloquearía un `.exe` local (es lo que pasó con torchcodec: `WinError 4551`). Desactivarlo es **irreversible sin reinstalar Windows**. Se usa el build de [lemonade-sdk/whisper.cpp-rocm](https://github.com/lemonade-sdk/whisper.cpp-rocm), que sí pasa el filtro.
- **pyannote recibe el audio en memoria**, como `{'waveform', 'sample_rate'}`, porque torchcodec (su backend por defecto) está bloqueado por SAC.
- **Decodificación con PyAV, no con el binario ffmpeg.** PyAV trae FFmpeg embebido y sí pasa SAC; además evita una dependencia externa.

---

## 5. Decisiones bloqueadas

Nunca cambiarlas sin confirmación explícita del usuario.

| ID | Estado | Decisión |
|----|--------|----------|
| D1 | ✅ Cerrada | FastAPI + React/Vite/TS. Dev: front `:6023`, back `:6024`. Sin login ni RBAC — herramienta local de un solo usuario. |
| D2 | ✅ Cerrada | **Sin base de datos.** Jobs en memoria; transcripts en `backend/data/outputs/`. PostgreSQL/SQLite no aportan nada aquí. |
| D3 | ✅ Cerrada | Transcripción con **whisper.cpp + Vulkan** (GPU). Fallback automático a faster-whisper CPU si falta binario o modelo. |
| D4 | ✅ Cerrada | Diarización con **pyannote.audio 4.x** en CPU, modelo `speaker-diarization-3.1`. Sin ruta GPU posible: es PyTorch. |
| D5 | ✅ Cerrada | La librería debe ser pyannote **4.x**: la 3.x usa `torchaudio.AudioMetaData`, eliminado en torchaudio ≥ 2.11. Versión de librería ≠ versión de modelo. |
| D6 | ✅ Cerrada | El **número de hablantes va por petición** (`num_speakers` en el formulario), no en configuración global. Cambia con cada audio; obligar a editar `.env` y reiniciar entre grabaciones no tiene sentido. |
| D7 | ✅ Cerrada | Salida Fase 1 = `.txt` con formato `[hh:mm:ss] Speaker N: texto`, pensado para alimentar un LLM. |
| D8 | 🔶 Pendiente | Modelo definitivo: `large-v3-q5_0` (1 GB) está en uso, pero hay una posible regresión de puntuación sin confirmar. Ver DT-1. |
| D9 | 🔶 Pendiente | Fase 2 — integración con Ollama (ya instalado, `qwen2.5:3b` y `gemma4:e2b` disponibles). Sin definir el prompt ni el flujo. |
| — | Asumido | Idioma por defecto español (`WHISPER_LANGUAGE=es`), todo el procesamiento local, nada sale a internet salvo la descarga inicial de modelos. |

---

## 6. Modos de invocación (prefijos de mensaje)

### `plan:` — Modo Planner (no ejecuta, solo especifica)

Claude actúa exclusivamente como Planner: genera un archivo `.md` en `Planes/` con la especificación completa para que un **agente externo sin acceso al repo ni contexto previo** lo ejecute al pie de la letra.

Reglas clave:
1. Solo genera el plan, nunca ejecuta. Cero ediciones a código.
2. El plan es 100% autocontenido (el executor no ve conversaciones ni git previos).
3. Rutas **absolutas** siempre.
4. Código de referencia completo para cada archivo a crear.
5. Contexto del proyecto inline (stack, estructura, convenciones, env vars).
6. Dependencias explícitas + check de verificación ejecutable.
7. Criterios de aceptación verificables (tabla comando → resultado esperado).
8. Decisiones cerradas: el executor no decide nada.
9. Secciones: Contexto → Objetivo → Prerequisitos → Inventario archivos → Especificación → Orden ejecución → Reglas no negociables → Validaciones → Fuera de alcance.
10. Naming: `Planes/plan_[ID_TAREA]_[fecha].md`.
11. Mostrar ruta + resumen de 5 líneas → esperar "¿Aprobado?".

Prompt estándar para el executor:
```
Eres un agente EXECUTOR. Lee completo el plan indicado y ejecútalo AL PIE DE LA LETRA.
Reglas: CERO modificaciones. Orden secuencial. Si falla, DETENTE. Reporta: ✅/❌ Paso N.
Al final: archivos tocados + "¿Hago commit?"
```

### `backup:` — Modo Backup

Ejecuta `scripts/backup.ps1` (pendiente de crear, ver DT-5). Naming con timestamp (`backup_{YYYYMMDD_HHMM}.zip`). Respalda Tier 1 irrecuperables (`backend/.env`) y Tier 2 caros de regenerar (`backend/data/outputs/`).

**No respaldar** `vendor/` ni `models/`: son ~1 GB de artefactos redescargables desde sus fuentes originales.

Incluir `MANIFEST.md` dentro del zip con git HEAD, branch y receta de restauración.

---

## 7. Directiva de auditoría previa antes de escribir un plan

🔴 **Nunca entregar un plan "v1 improvisado" para mejorarlo en "v2" cuando el usuario detecte fallos.** El plan entregado debe ser ya un v2 auditado.

Si la tarea toca >3 archivos, introduce primitivos/hooks/utils nuevos, modifica archivos compartidos, o cambia contratos entre capas → ejecutar antes:
1. Grep de archivos similares existentes (confirmar convención).
2. Read completo del archivo a modificar (no de memoria).
3. Verificar paths de imports contra archivos vecinos.
4. Leer configs relevantes (`vite.config.ts`, `package.json`, `tsconfig.json`, `requirements.txt`, `.env.example`).
5. Cruzar contra la deuda técnica (§ 10).
6. Cruzar contra reglas duras (§ 9) y restricciones del entorno (§ 4).

Si la auditoría revela un bloqueante (decisión bloqueada afectada, riesgo crítico) → detener y escalar **antes** de escribir el plan.

Anti-patrones prohibidos: plan v1 "rápido" a sabiendas, asumir paths/configs de memoria, "esto probablemente funciona, lo confirma el typecheck", esperar a que el usuario pida "aplica el flujo profesional".

---

## 8. Flujo profesional de ejecución (6 pasos)

Antes de cualquier tarea no trivial:

**Mapeo → Auditoría → Diagnóstico → Propuesta → Aplicación → Verificación**

- No saltear pasos. Propuesta completa antes de aplicar.
- Si un hallazgo afecta una decisión bloqueada → detener y escalar.
- **Verificación = tests verdes + typecheck verde + EJECUCIÓN REAL DEL PIPELINE.** Para este proyecto, "verificado" significa haber procesado audio de verdad, no solo que importen los módulos.

---

## 9. Reglas duras

- **R1 — No modificar infraestructura compartida sin ADR.** `requirements.txt`, `package.json`, `vite.config.ts`, `tsconfig.json`. Si una librería falla al instalar: diagnosticar, no aplicar atajos. Si no se resuelve en 15 min → detener y escalar.
- **R2 — Nunca cargar dos modelos a la vez.** Con 7,9 GB de RAM, romper la carga secuencial de `pipeline.py` lleva la máquina a swap y el proceso se vuelve más lento que el cómputo. Todo motor nuevo debe liberar en `__exit__`.
- **R3 — "Tests verdes" ≠ "funciona".** Los tests no cargan modelos ni tocan la GPU. Los tres fallos reales de este proyecto (pyannote 3.x incompatible, licencia de `community-1` faltante, `DiarizeOutput` sin `itertracks`) solo aparecieron al ejecutar el pipeline completo.
- **R4 — Si un fix se acumula >2 iteraciones sin resolver el bug, detener y revertir** al último estado bueno conocido. No seguir parchando.
- **R5 — Verificar acceso, no metadatos.** Que `model_info()` de HuggingFace responda **no** significa que haya acceso a los archivos: en un repo *gated* los metadatos son públicos y los archivos no. La única comprobación válida es descargar un archivo real (`scripts/check_setup.py` lo hace así).
- **R6 — Nunca compilar binarios localmente.** Smart App Control los bloquea y desactivarlo es irreversible. Si hace falta un binario nuevo, buscar build precompilado de fuente reputada.
- **R7 — Un cambio de configuración por experimento.** Al comparar rendimiento o calidad, variar **una** cosa por vez. Cambiar motor + modelo + cuantización + dispositivo a la vez impide atribuir el resultado (pasó al integrar la GPU).

---

## 10. Deuda técnica

| # | Item | Resolver en | Detalle |
|---|------|-------------|---------|
| DT-1 | Regresión de puntuación con `large-v3-q5_0` | Próxima sesión | La salida perdió mayúsculas y puntuación frente a `small`. Causa sin confirmar: se cambiaron motor + modelo + cuantización + dispositivo a la vez. Contraevidencia: el mismo modelo sí produjo `¡Gracias por ver el video!` en otra prueba. Pendiente: `benchmark_engines.py` sobre audio real (requiere `KEEP_UPLOADS=true`). |
| DT-2 | Ratio real sin medir | Próxima sesión | Todas las cifras salen de audio sintético, que genera pocos tokens y subestima el decoder. Único dato real: 23 s de audio → 27 s (1,2×), dominado por costes fijos de carga. |
| DT-3 | Fase 2 — Ollama | Tras cerrar DT-1 | Entregar el `.txt` a un LLM local. Ollama instalado con `qwen2.5:3b` y `gemma4:e2b`. Nota: sus variables persistidas fuerzan CPU (`OLLAMA_LLM_LIBRARY=cpu`, `OLLAMA_VULKAN=0`) — se puede subir a GPU, ya está probado. |
| DT-4 | Diarización posiblemente cuello de botella | Tras DT-2 | Con la transcripción en GPU, pyannote en CPU puede dominar el tiempo. Reemplazo natural: `sherpa-onnx` (ONNX Runtime, más liviano) implementando `DiarizationEngine`. |
| DT-5 | `scripts/backup.ps1` | Setup | No existe. Necesario para que el modo `backup:` funcione. |
| DT-6 | Token HF expuesto | Cuando convenga | El `HF_TOKEN` se pegó en una conversación. Es de solo lectura, pero conviene regenerarlo en https://huggingface.co/settings/tokens. |
| ~~DT-7~~ | ~~Sin control de versiones~~ | ✅ Cerrada 2026-07-23 | Repo en https://github.com/jaguez40-star/audiotracker · rama `main` · commit inicial `a28f79d`. Identidad configurada solo a nivel de repo. |

Cualquier `# TODO[DT-x]:` en código → entrada espejo en esta tabla. Al cerrar, eliminar de tabla + referenciar commit.

---

## 11. Reglas operativas de commits y ramas

1. **Cada commit referencia su tarea por ID:** `feat(T1.4): añadir motor whisper.cpp`.
2. Commits, mensajes y ramas en español.
3. **Nunca `git add -A`** en commits acumulados grandes: formatear antes de stagear, stagear por bloque lógico.
4. **Nunca versionar** `vendor/`, `models/`, `backend/data/`, `backend/.env`, `backend/venv/`, `node_modules/`. Ya están en `.gitignore`.
5. Si una tarea se bloquea: marcar `🔴 BLOCKED`, pasar a la siguiente, no atascarse >1h.
6. Nunca saltar hooks (`--no-verify`) ni firmar/omitir sin pedido explícito.

---

## 12. Trampas conocidas del stack

Cosas que costaron tiempo descubrir. Consultar antes de depurar lo mismo dos veces.

| Síntoma | Causa | Solución |
|---|---|---|
| `AttributeError: torchaudio has no attribute AudioMetaData` | pyannote 3.x con torchaudio ≥ 2.11 | Usar pyannote ≥ 4.0 |
| `DiarizeOutput object has no attribute itertracks` | pyannote 4 envuelve el `Annotation` | `resultado.speaker_diarization` (resuelto por duck typing en `pyannote_engine.py`) |
| `GatedRepoError 403` tras transcribir | Falta licencia de `speaker-diarization-community-1` | pyannote 4 saca de ahí el PLDA aunque se use el modelo 3.1. Son **tres** repos, no dos |
| `WinError 4551` al cargar una DLL | Smart App Control | No compilar local; usar binarios con reputación |
| `Input should be a valid integer` al arrancar | Variable vacía en `.env` (`MIN_SPEAKERS=`) | Resuelto con `_blank_to_none` en `core/config.py` |
| Un hablante se parte en dos | Pausa larga interpretada como cambio de voz | Indicar `num_speakers` al subir |
| La insignia del front muestra motor viejo | `staleTime` alto sin refetch | Resuelto: `refetchInterval` en la query de health |
| Whisper inventa texto ("Subtítulos por Amara.org") | Alucinación sobre audio sin voz | Esperable con silencios o ruido; no es un bug |

---

## 13. Bitácora de sesiones

| Fecha | ID | Descripción | Archivos | Commits |
|-------|----|-------------|----------|---------|
| 2026-07-23 | SETUP-0 | Diseño inicial y scaffold completo: backend FastAPI con motores tras protocolos, frontend React página única, carga secuencial de modelos por límite de 8 GB RAM. Descarte verificado de CUDA/ROCm/DirectML para gfx1031. | `backend/**`, `frontend/**`, `README.md`, `Start_*.bat`, `.gitignore` | — |
| 2026-07-23 | FIX-1 | Tres fallos de integración detectados solo al ejecutar el pipeline real: pyannote 3.x incompatible con torchaudio 2.11 (subida a 4.0.7), tercer repo con licencia (`community-1`) del que pyannote 4 saca el PLDA, y `DiarizeOutput` en vez de `Annotation`. Añadido preflight que descarga archivos reales en vez de fiarse de `model_info()`. | `engines/pyannote_engine.py`, `requirements.txt`, `scripts/check_setup.py`, `.env.example` | — |
| 2026-07-23 | FIX-2 | Arranque roto por variables vacías en `.env`: `MIN_SPEAKERS=` no parsea como `int \| None`. Resuelto con `_blank_to_none`, para que dejar una variable en blanco signifique "ausente". Migrado `on_event` a `lifespan`. | `core/config.py`, `main.py` | — |
| 2026-07-23 | UX-1 | `num_speakers` movido de configuración global a parámetro por petición (D6) tras un falso hablante causado por una pausa larga. Verificado: el mismo audio pasó de 2 hablantes a 1 correcto. Añadida medición `processing_seconds`/`speed_ratio`, que no existía. | `routes.py`, `jobs.py`, `schemas.py`, `pipeline.py`, `components/SpeakerHint.tsx`, `api/client.ts` | — |
| 2026-07-23 | GPU-1 | Vulkan verificado estable con Ollama (100% GPU, 51 vs 9,5 tok/s). Integrado whisper.cpp + Vulkan con binario precompilado de lemonade-sdk: evita compilar y evita desactivar Smart App Control, que es irreversible. Modelo `large-v3` ahora viable. Benchmark: **10,4× GPU vs CPU** con modelo idéntico. | `engines/whispercpp_engine.py`, `engines/__init__.py`, `engines/base.py`, `core/config.py`, `pipeline.py`, `routes.py`, `schemas.py`, `scripts/benchmark_engines.py`, `tests/test_whispercpp.py`, `vendor/`, `models/` | — |
| 2026-07-23 | UX-2 | La insignia del frontend mostraba `small · cpu` con la GPU ya activa: la consulta de estado tenía `staleTime` alto sin refresco. Corregido con `refetchInterval`. | `App.tsx`, `api/client.ts` | — |
| 2026-07-23 | DOC-1 | `CLAUDE.md` adaptado desde el proyecto `sellweb`: conservadas las secciones de método (§ 1, 6, 7, 8, 11, 13) y reemplazados stack, arquitectura y dominio. Añadidas § 4 (restricciones de entorno verificadas) y § 12 (trampas conocidas) para no repetir la investigación de esta sesión. | `CLAUDE.md` | — |
| 2026-07-23 | DOC-2 | Estructura del proyecto verificada contra el disco y documentada al completo. Añadida § 3-bis con las funcionalidades operativas. `.gitignore` corregido: no cubría `vendor/`, `models/` ni `*.tsbuildinfo`, contradiciendo la regla 4 de § 11. | `CLAUDE.md`, `.gitignore` | — |
| 2026-07-23 | GIT-1 | `git init` + commit inicial + publicación en GitHub (cierra DT-7). Auditoría previa al commit: verificado archivo por archivo que `.env`, `models/`, `vendor/`, `venv/` y `data/` quedan excluidos, y que el token no aparece en el contenido preparado. 50 archivos, 6.489 líneas. | todo el proyecto | `a28f79d` |

### Estado al cierre de la sesión

| Indicador | Valor |
|---|---|
| Tests | 36 pasan |
| Typecheck frontend | limpio |
| Build producción | 78 KB gzipped |
| Preflight | todo en verde |
| Pipeline end-to-end | verificado con GPU |
| Transcripción | `large-v3` en GPU · 0,30× sobre audio sintético |
| Dato real disponible | 23 s de audio → 27 s (1,2×), dominado por costes fijos |
| Bloqueante para cerrar DT-1 | falta audio real con `KEEP_UPLOADS=true` |
