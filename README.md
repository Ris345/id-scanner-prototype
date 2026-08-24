# ID Scanner

Scan government-issued IDs (driver's licenses, state IDs, passports) with a camera or photo
gallery and extract structured fields: name, date of birth, address, ID number, expiry date,
issue date, sex, state, and document type.

OCR runs **locally** — a PDF417 barcode decoder, an MRZ reader, and a vision LLM on your own
hardware. Scanned IDs are not sent to a third-party OCR service unless the local pipeline is
completely unreachable.

## Architecture

| Tier | Stack | Port | Runs in |
|---|---|---|---|
| Frontend | Expo (React Native) — iOS, Android, web | 8081 | host |
| Backend | Node.js / Express | 3001 | Docker |
| OCR microservice | Python / Flask | 3002 | Docker |
| Vision model | Ollama + `llama3.2-vision:11b` | 11434 | **host, not Docker** |

Ollama stays on the host because a container cannot reach Apple's Metal GPU — an 11B vision
model inside Docker runs CPU-only, taking minutes per scan instead of ~30–60s. The Python
service reaches it via `host.docker.internal`.

Inside the compose network, Node reaches Python at `http://python-ocr:3002` — not `localhost`,
which inside a container is the container itself. Both ports are still published, so host tools
and the app keep using `localhost`.

### OCR pipeline

Stages run in order of trustworthiness and stop as soon as the result is complete:

```
1. PDF417 barcode (zxing-cpp)  → confidence 1.00   the card's own machine-readable data
2. MRZ (PassportEye)           → confidence 0.98   passports, checksum-verified
3. llama3.2-vision:11b         → confidence 0.70   reads the card image directly
4. Tesseract                   → raw text only     safety net, never fields
```

AWS Textract is a **crash-only fallback** in Node — it fires only if the Python service is
unreachable entirely, never on a low-confidence result. It is optional; leave the AWS
credentials unset and the pipeline simply has one less fallback.

## Setup

### Prerequisites

```bash
brew install tesseract zbar        # system OCR + barcode libs
ollama pull llama3.2-vision:11b    # ~8GB, one time
```

Docker Desktop is required for the backend services.

### 1. Ollama (host, separate terminal)

```bash
OLLAMA_HOST=0.0.0.0 ollama serve
```

`0.0.0.0` matters — the default binds to loopback only, which the Docker container cannot
reach. Pre-warm before the first scan so it doesn't time out loading 8GB of weights:

```bash
ollama run llama3.2-vision:11b "hi"
```

### 2. Backend stack

`docker-compose.yml` lives in `backend/`, not `backend/python-easy-ocr/`.

```bash
cd backend && docker-compose up -d
docker-compose logs -f              # tail both services
docker-compose logs -f python-ocr   # just the [PY] pipeline
```

The stack is healthy in ~40s. Check it:

```bash
curl localhost:3001    # should report python_ocr: "available"
```

> **Never run `docker-compose down --remove-orphans` from `backend/`.** The compose project
> name is pinned to `id-scanner` for a reason — without it Compose derives the name from the
> directory (`backend`) and will adopt and destroy identically-named services from any other
> project on your machine that also has a `backend/` dir. This has already caused one incident.

### 3. Frontend

Two options.

**Browser scanner (recommended for phone testing)** — no app install, works from any network:

```bash
ngrok http 3001
# then open <ngrok-url>/scan on the phone
```

ngrok's HTTPS is what makes `getUserMedia` work; browsers block camera access over plain HTTP.
The free ngrok URL changes on every restart.

**Expo:**

```bash
npm install
npx expo start
```

Press `w` for browser, or scan the QR with Expo Go. In dev the app resolves the backend from
Metro's own host, so a physical phone must be on the same network. Under `expo start --tunnel`
that fails — only port 8081 is forwarded — so set `EXPO_PUBLIC_API_URL` to the backend's own
public URL for tunnelled runs.

## Configuration

AWS credentials are **optional** (Textract crash fallback only). If you want them, create
`.env` in the **project root** — not in `backend/`; `server.js` loads `../.env` relative to
itself:

```
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1
```

Python service environment flags:

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_ENABLED` | `true` | Set `false` to skip the vision stage |
| `OLLAMA_MODEL` | `llama3.2-vision:11b` | Override the model |
| `OLLAMA_HOST` | `host.docker.internal` | Use `localhost` when running outside Docker |
| `OLLAMA_TIMEOUT` | `180` | Seconds; must stay under Node's `SCAN_TIMEOUT_MS` |
| `OLLAMA_KEEP_ALIVE` | `30m` | Holds the model resident; Ollama otherwise evicts after ~5 min and the reload outlasts the request timeout |

## API

```
GET  /           Health check — reports Python and Textract availability
GET  /scan       Browser scanner UI
POST /api/scan   { image: <base64 or data URI>, side?: 'front' | 'back' }
```

`side` is an optional hint that skips irrelevant stages — `back` skips MRZ, `front` skips the
barcode.

Each field comes back as a structured object, not a plain string:

```json
{
  "name":        { "value": "JOHN SMITH", "confidence": 0.70, "source": "ollama" },
  "dateOfBirth": { "value": "01/15/1990", "confidence": 1.0,  "source": "barcode" }
}
```

## Production

- **Live demo:** https://incomparable-donut-edf69d.netlify.app/

> **The deployed backend is not the pipeline described above.** The hosted Render service runs
> Node only — no Python microservice and no Ollama — so it reports
> `python_ocr: "unavailable"` and every scan there falls through to AWS Textract. It is also
> pinned to an older build. `app/utils/ocr.ts` points production builds at it, so a production
> build gets Textract-only behaviour. Treat the hosted demo as a Textract demo; the local-first
> pipeline is what this repo develops against.

Redeploy the frontend as a static web export:

```bash
npx expo export --platform web   # outputs to dist/
```

Use the site's bare URL when linking a Netlify deploy. A URL carrying a `<hash>--` prefix is a
per-deploy preview link and 404s once that deploy is superseded — which is exactly how the
previous link in this file broke.
