# audio_track

Transcripción local de audio con identificación de hablantes. Se sube un archivo
(mp4, wav, mp3…), el sistema lo transcribe, separa quién habla, y entrega un
`.txt` con el formato `Speaker N: texto`.

Todo el procesamiento ocurre en la máquina. Nada sale a internet salvo la
descarga inicial de los modelos.

**Fase 1** — transcripción y diarización. La salida `.txt` está pensada para
alimentar un LLM local vía Ollama en la Fase 2.

---

## Stack

| Capa | Tecnología |
|---|---|
| Backend | Python 3.12 · FastAPI 0.139 · Pydantic v2 · structlog |
| **Transcripción** | **whisper.cpp + Vulkan · modelo `large-v3` · GPU** |
| Transcripción (respaldo) | faster-whisper 1.2 · modelo `small` int8 · CPU |
| Diarización | pyannote.audio 4.0 · `speaker-diarization-3.1` · CPU |
| Frontend | React 19 · TypeScript 5.7 · Vite 6 · Sass Modules · TanStack Query 5 |

---

## GPU: transcripción en la RX 6700 XT vía Vulkan

La transcripción corre en la GPU. Llegar ahí requirió descartar todas las rutas
convencionales, porque la **RX 6700 XT es RDNA2 (gfx1031)** y casi ningún stack
de ML la cubre en Windows:

| Ruta | Resultado | Cómo se verificó |
|---|---|---|
| CUDA | ❌ No aplica | No hay GPU NVIDIA |
| PyTorch + ROCm Windows | ❌ Solo gfx1100/1101/1200/1201 (RDNA3/4) | Matriz de compatibilidad de AMD |
| CTranslate2 + ROCm | ❌ Compila gfx1030 y gfx1100+, no gfx1031 | Lista de targets de su CI |
| `HSA_OVERRIDE_GFX_VERSION` | ❌ Exclusivo de Linux | No existe en Windows |
| torch-directml | ❌ Abandonado (sept 2024, torch 2.4.1) | Última versión en PyPI |
| **Vulkan** | ✅ **Funciona** | Probado con Ollama: **100% GPU, 51 vs 9,5 tok/s** |

Ollama sirvió de banco de pruebas porque usa la misma librería `ggml` que
whisper.cpp y trae backends ROCm y Vulkan ya compilados. Su detector rechaza
ROCm con `hipGetDeviceCount failed: 100` — AMD no distribuye runtime HIP para
RDNA2 en Windows — pero con Vulkan cargó el modelo **entero en la GPU** de forma
estable. Los BSODs que documentaba el proyecto ProdIA eran de un driver anterior;
el 26.6.4 los resolvió.

### El binario

whisper.cpp **oficial no publica binarios Vulkan para Windows** (issue #3673,
abierta desde feb 2026). Usamos los de
[lemonade-sdk/whisper.cpp-rocm](https://github.com/lemonade-sdk/whisper.cpp-rocm),
que sí los compila y sigue a upstream automáticamente.

Esto además esquiva un obstáculo real: **Smart App Control está activo en este
equipo** y bloquea binarios sin firma ni reputación — es lo que impidió cargar
las DLLs de torchcodec (`WinError 4551`). Un `whisper.exe` compilado localmente
habría caído en ese filtro; el binario de GitHub pasa. Compilar desde fuente
habría obligado a desactivar Smart App Control, que es **irreversible sin
reinstalar Windows**.

### Reparto del trabajo

```
Transcripción (whisper.cpp)  →  GPU vía Vulkan   ⚡ large-v3
Diarización   (pyannote)     →  CPU              (PyTorch, sin ruta AMD)
```

La diarización se queda en CPU sin alternativa: pyannote es PyTorch, y PyTorch no
tiene build funcional para esta tarjeta. Por eso el pipeline no acelera de forma
uniforme.

Beneficio secundario nada menor con 8 GB de RAM: el modelo de transcripción vive
en la **VRAM del subproceso**, no en la memoria del backend, dejando los 8 GB
libres para pyannote.

### Por qué se pudo cambiar sin reescribir nada

Los motores viven detrás de los protocolos `TranscriptionEngine` y
`DiarizationEngine` en [base.py](backend/app/engines/base.py). Añadir la GPU
consistió en escribir
[whispercpp_engine.py](backend/app/engines/whispercpp_engine.py) cumpliendo el
mismo contrato. El pipeline, el job store, la API y el frontend no cambiaron.

El selector está en [engines/\_\_init\_\_.py](backend/app/engines/__init__.py):
si `TRANSCRIPTION_ENGINE=whispercpp` pero falta el binario o el modelo, avisa por
log y **degrada a CPU** en vez de impedir el arranque.

---

## Restricción WDAC del equipo

La política corporativa **WDAC** bloquea DLLs sin firmar, y eso condiciona dos
decisiones que de otro modo parecerían arbitrarias:

**1. No usamos el binario de ffmpeg.** `torchcodec` — el backend de audio por
defecto de pyannote 4 — falla al cargar sus DLLs con
`WinError 4551: Una directiva de Control de aplicaciones bloqueó este archivo`.
Instalar FFmpeg no lo resolvería: el bloqueo es sobre las DLLs de torchcodec.

En su lugar:

- **Decodificación** vía PyAV (`av`), que viene con faster-whisper y trae sus
  propias librerías FFmpeg — verificado, no lo bloquea WDAC.
- **Lectura para pyannote** vía `soundfile`/libsndfile, pasándole el audio ya
  cargado en memoria como `{'waveform', 'sample_rate'}`. Es la vía que el propio
  pyannote recomienda cuando torchcodec no está disponible, y además evita
  releer el archivo.

**2. `pyannote.audio` debe ser 4.x.** La 3.x usa `torchaudio.AudioMetaData`,
eliminado en torchaudio ≥ 2.11, y falla al importar. El modelo sigue siendo
`speaker-diarization-3.1` — la versión de la librería y la del modelo son cosas
distintas.

---

## Instalación

### 1. Backend

```powershell
cd backend
python -m venv venv
venv\Scripts\pip install -r requirements.txt
```

La instalación descarga PyTorch y pesa varios GB. Toma su tiempo.

### 2. Token de HuggingFace (obligatorio)

pyannote requiere aceptar la licencia de sus modelos. Con **la misma cuenta**:

1. Crear un token tipo *read* en https://huggingface.co/settings/tokens
2. Aceptar las condiciones en **los tres** repositorios:
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/speaker-diarization-community-1

Los tres son de aprobación automática e instantánea.

El tercero no es evidente y no aparece en la mayoría de los tutoriales:
pyannote 4.x descarga de ahí el componente PLDA (`plda/xvec_transform.npz`)
aunque se esté usando el modelo `3.1`. Sin aceptarlo, el pipeline falla con
`GatedRepoError 403` al construirse — después de que la transcripción ya corrió.

> **Verificar antes de procesar:** que `model_info()` responda no significa que
> haya acceso. Los metadatos de un repo gated son públicos; los archivos no. La
> comprobación válida es descargar un archivo real.

```powershell
copy .env.example .env
```

Y completar `HF_TOKEN=` en `backend\.env`.

### 3. Frontend

```powershell
cd frontend
npm install
```

### 4. Verificar

```powershell
cd backend
venv\Scripts\python -m pip install -r requirements-dev.txt
venv\Scripts\python -m pytest
```

Las pruebas cubren la atribución de hablantes y los contratos de la API. No
cargan modelos, así que corren en menos de un segundo.

---

## Uso

Dos terminales:

```
Start_Back.bat     →  http://127.0.0.1:6024
Start_Front.bat    →  http://localhost:6023
```

Abrir http://localhost:6023, arrastrar un audio, esperar.

La primera ejecución descarga los modelos (~2 GB entre Whisper y pyannote) y es
notablemente más lenta que las siguientes.

---

## Rendimiento

Medido con `scripts/benchmark_engines.py` sobre 60 s de audio, en un i5-10400
(6 núcleos) + RX 6700 XT:

| Configuración | Tiempo | Ratio |
|---|---|---|
| faster-whisper `small` · CPU | 5,7 s | 0,10× |
| **whisper.cpp `large-v3` · GPU** | **18,3 s** | **0,30×** |
| whisper.cpp `large-v3` · CPU | 190,2 s | 3,17× |

Dos lecturas distintas de la misma tabla:

**Vulkan rinde.** Mismo binario y mismo modelo, cambiando solo el dispositivo:
190,2 s → 18,3 s, **10,4× de aceleración**.

**La GPU no compró velocidad, compró calidad.** El modelo `small` en CPU sigue
siendo más rápido que `large-v3` en GPU. Lo que la GPU hace posible es usar el
modelo grande —muy superior en español, sobre todo con acentos, nombres propios y
ruido de fondo— a una velocidad utilizable. En CPU ese modelo es inviable.

> Estas cifras salen de audio sintético, que genera pocos tokens y por tanto
> apenas ejercita el decoder, que es donde más ayuda la GPU. Con habla real es
> esperable que `large-v3` en GPU mantenga o amplíe su ventaja, mientras `small`
> en CPU se degrada. Pendiente de medir.

### Palancas si hace falta ir más rápido

En orden de impacto:

1. **Medir dónde se va el tiempo.** Con la GPU haciendo la transcripción, es
   probable que la diarización en CPU sea ahora el cuello de botella. El campo
   `speed_ratio` de cada job da el dato.
2. **Indicar el número de hablantes** al subir el audio. Mejora la calidad de la
   diarización y le ahorra trabajo.
3. **Cambiar la diarización a `sherpa-onnx`** (ONNX Runtime, mucho más liviano en
   CPU) implementando `DiarizationEngine`.
4. **Bajar a `medium`** en `WHISPERCPP_MODEL` si `large-v3` resulta excesivo.

---

## Configuración

Todo se ajusta en `backend\.env` sin tocar código. Ver
[.env.example](backend/.env.example) para la lista completa.

Lo que más mueve la aguja:

| Variable | Efecto |
|---|---|
| `WHISPER_MODEL` | `tiny`/`base`/`small`/`medium`/`large-v3`. `small` es el balance para 8 GB de RAM |
| `WHISPER_LANGUAGE` | Fijarlo (`es`) es más rápido y preciso que la detección automática |
| `MIN_SPEAKERS` / `MAX_SPEAKERS` | Respaldo global. Normalmente conviene usar el selector de la interfaz, que va por archivo |
| `KEEP_UPLOADS` | `true` conserva el audio original tras procesar |

---

## API

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/api/health` | Estado y configuración activa |
| `POST` | `/api/transcribe` | Sube un audio, devuelve un job. Acepta `num_speakers` opcional. Procesa en background |
| `GET` | `/api/jobs` | Lista de jobs |
| `GET` | `/api/jobs/{id}` | Estado y progreso |
| `GET` | `/api/jobs/{id}/result` | Transcript completo en JSON |
| `GET` | `/api/jobs/{id}/download` | Descarga el `.txt` |

Documentación interactiva en http://127.0.0.1:6024/docs

---

## Notas de arquitectura

**Carga secuencial de modelos.** El equipo tiene 8 GB de RAM y los dos modelos
juntos no caben con holgura. El pipeline los carga y libera de a uno mediante
context managers, manteniendo el pico alrededor de 2 GB. Ver
[pipeline.py](backend/app/pipeline.py).

**Normalización de audio.** Ambos motores decodifican con backends distintos, y
el de pyannote es débil con contenedores mp4/m4a. Todo se convierte primero a WAV
PCM 16 kHz mono usando PyAV, que ya viene con faster-whisper — así no hace falta
instalar el binario de ffmpeg por separado. Ver [audio.py](backend/app/audio.py).

**Atribución de hablantes.** Las fronteras de Whisper y las de pyannote no
coinciden. Cada segmento de texto se asigna al hablante con mayor solape temporal
acumulado, no al primer turno que toca. Luego se renumera por orden de aparición
para que `Speaker 1` sea siempre quien habla primero. Ver
[merge.py](backend/app/merge.py).

**El número de hablantes va por petición, no por configuración.** Es el
parámetro que más afecta la calidad de la diarización: sin él, una pausa larga o
un cambio de entonación bastan para que pyannote invente un hablante que no
existe. Como ese número cambia con cada grabación, vive en el formulario de
subida (`num_speakers`) y no en el `.env` — obligar a editar un archivo y
reiniciar el servidor entre audios no tendría sentido. Los valores del `.env`
quedan solo como respaldo cuando la petición no trae dato.

**Sin base de datos.** Los jobs viven en memoria y se pierden al reiniciar; los
transcripts quedan en `backend\data\outputs`. Es una herramienta local de un solo
usuario — PostgreSQL, auth y RBAC no aportarían nada aquí.
