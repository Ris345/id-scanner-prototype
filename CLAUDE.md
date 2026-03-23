# ID Scanner Prototype

Government ID document scanning app that extracts structured data from photos of driver's licenses, passports, and national IDs.

## Architecture

Three-tier system:
- **Frontend**: Expo (React Native) — `app/`
- **Backend**: Node.js/Express on port 3001 — `backend/server.js`
- **Python OCR microservice**: Flask on port 3002 — `backend/python-easy-ocr/app.py`

## Dev Commands

```bash
# Backend (Node — nodemon auto-restarts on save)
cd backend && npm run dev

# Python microservice (Docker — always use this)
cd backend/python-easy-ocr && docker-compose up -d
docker-compose logs -f  # tail logs

# Copy app.py change into running container (no rebuild needed for Python-only changes)
docker cp app.py python-ocr-1:/app/app.py && docker restart python-ocr-1

# Full rebuild (needed when requirements.txt or Dockerfile changes)
cd backend/python-easy-ocr && docker-compose down --rmi all && docker-compose build --no-cache && docker-compose up -d && docker-compose logs -f

# Ollama (run in separate terminal before scanning)
ollama serve  # if not already running — check with: lsof -i :11434

# Frontend
npx expo start
```

## OCR Pipeline

**Priority order**: barcode > MRZ > llama3.2-vision > Textract (crash fallback only)

```
Image
  ↓
1. PDF417 barcode decode       → if found + parsed → return (confidence 1.0)
  ↓
2. MRZ (PassportEye)           → if found + checksum passes → return (confidence 0.98)
  ↓
3. llama3.2-vision:11b (Ollama) → sends image directly, returns structured JSON fields
```

- `POST /api/scan` → Python microservice always first
- **Ollama receives the image** — vision model reads the ID card directly, no OCR pre-step
- **Textract**: only fires in Node.js if the Python service throws entirely
- Optional `side: 'front' | 'back'` in request body skips irrelevant stages
- Document classifier (aspect ratio) gates which stages run: `dl_or_stateid` | `passport` | `unknown`
- Node.js timeout: 120s (llama3.2-vision:11b is slow on first load, ~30-60s)

## Output Schema

Each field is a structured object — NOT a plain string:
```json
{
  "name":        { "value": "JOHN SMITH", "confidence": 0.92, "source": "llama-vision" },
  "dateOfBirth": { "value": "01/15/1990", "confidence": 1.0,  "source": "barcode" }
}
```
`source` values: `"barcode"` | `"mrz"` | `"llama-vision"` | `"textract"`

**server.js unwraps with**: `const fv = key => (f[key] && f[key].value) || null`

Response also includes `warnings: ["glare_detected"]` if > 15% of image is blown out.

## Output Fields

`name`, `dateOfBirth`, `address`, `idNumber`, `expiryDate`, `issueDate`, `sex`, `state`, `documentType`

## Key Files

| File | Purpose |
|------|---------|
| `backend/server.js` | Express API, Python call, Textract crash fallback, `logScanResult()` |
| `backend/python-easy-ocr/app.py` | OCR pipeline — barcode + MRZ + docTR raw text + Ollama structuring |
| `backend/python-easy-ocr/Dockerfile` | Pre-bakes docTR models at build time |
| `backend/python-easy-ocr/requirements.txt` | Python deps |
| `app/utils/ocr.ts` | `scanID(uri, side?)` — frontend HTTP client |
| `app/utils/idParser.ts` | `ParsedID` interface |
| `app/scan.tsx` | Camera UI, capture/gallery, pinch-to-zoom |
| `app/form.tsx` | Verification form with editable fields |
| `app/context/ScanContext.tsx` | Cross-screen state for scanned data |

## Python Microservice Details

### Models
- **docTR**: `db_resnet50` (detection) + `parseq` (recognition) — used for raw text extraction only
- PaddleOCR removed entirely

### GPU detection
Auto-detects CUDA → Apple MPS → CPU at startup.

### Preprocessing
Upscale to 1200px min + CLAHE on LAB luminance channel. Applied before docTR.
Numpy array converted to PNG bytes before passing to `DocumentFile.from_images()`.

### Barcode
PDF417 via pyzbar, 3-strategy decode (full image → bottom-half crop → 2x upscale), AAMVA field mapping.

### MRZ
PassportEye (requires system Tesseract). Confidence 0.98 if checksum passes, 0.50 if not.

### Ollama vision
- Model: `llama3.2-vision:11b` (local, via `ollama serve`)
- URL: `host.docker.internal:11434` inside Docker, `localhost:11434` outside
- **Receives the raw image** — vision model reads the ID card directly
- Prompt instructs model to return structured JSON with all ID fields
- Returns fields with `confidence: 0.70, source: "llama-vision"`
- Silently falls through if Ollama not running
- Pre-warm before first scan: `ollama run llama3.2-vision:11b "hi"`
- Run Ollama with Metal GPU: `OLLAMA_HOST=0.0.0.0 ollama serve`

### Environment flags
- `OLLAMA_ENABLED=false` — disables Ollama structuring
- `OLLAMA_MODEL=gemma3:4b` — Ollama model
- `OLLAMA_HOST=host.docker.internal` — use `localhost` outside Docker

## Logging

All Python logs prefixed `[PY]` — distinct from `[Node]` in server.js.
Key prefixes to watch:
- `[PY] NEW REQUEST` — request banner with side and doc_class
- `[PY][Barcode]` — PDF417 decode result
- `[PY][MRZ]` — MRZ parse result
- `[PY][docTR]` — line count + raw text extracted
- `[PY][Ollama]` — field count returned
- `[PY] RESULT` — final table with confidence/source per field, `<-- MISSING` markers
- `[Node] SCAN RESULT` — Node-side view of final merged fields

## server.js Notes

- `scanWithPython()` unwraps `.value` from structured fields via `fv(key)` helper
- `confidence` passed through from Python response
- `logScanResult()` prints `[Node]` summary after every scan
- **Textract fires only if Python service throws** — not on low confidence
- Mobile/desktop platform split removed — single code path for all clients

## Current State

### What's working
- Pipeline: barcode → MRZ → llama3.2-vision:11b
- PaddleOCR, docTR spatial extraction, regex fallback all removed
- llama3.2-vision reads image directly — no OCR pre-step needed
- PII stays local — no external APIs
- Docker container clean, server.js Textract logic correctly nested in Python catch block
- Node.js timeout bumped to 120s for vision model load time

### Next steps
- Tune llama3.2-vision prompt — check raw model response in Docker logs first
- Test across ID types: NY permit, standard DL front, passport
- On-device barcode decode (React Native Vision Camera) — eliminates server round-trip for ~70% of DL scans
- Frontend form field mapping — `app/form.tsx` consuming structured schema

## Environment

AWS credentials in `backend/.env` are optional — Textract is crash fallback only.

```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
```

## System Dependencies (macOS)

```bash
brew install tesseract zbar
ollama pull gemma3:4b  # pull model once
```

## Docker

Models pre-baked at build time — no downloads at runtime:
- docTR: `db_resnet50` + `parseq`

System deps: `tesseract-ocr`, `libgl1`, `libglib2.0-0`, `libzbar0`.
