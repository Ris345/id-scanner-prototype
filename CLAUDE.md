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

# Full rebuild (needed when app.py or Dockerfile changes)
cd backend/python-easy-ocr && docker-compose down && docker-compose build --no-cache && docker-compose up -d && docker-compose logs -f

# Ollama (run in separate terminal before scanning)
ollama serve  # if not already running — check with: lsof -i :11434

# Frontend
npx expo start
```

## OCR Pipeline

**Priority order**: barcode > MRZ > docTR spatial > regex > Ollama > PaddleOCR > Textract (last resort)

- `POST /api/scan` → Python microservice always first
- **Regex**: gap-fills any field spatial extraction missed
- **Ollama** (`gemma3:4b`): triggered when name/dateOfBirth/idNumber still null after regex — sends raw OCR text with structured prompt, returns JSON
- **PaddleOCR**: triggered when docTR confidence < 0.75 for name/dateOfBirth/idNumber
- **Textract**: only fires if Python service crashes entirely
- Response `source` tag: `"doctr"` | `"doctr+ollama"` | `"doctr+paddle"` | `"doctr+passporteye+barcode"` | `"textract"`
- Optional `side: 'front' | 'back'` in request body skips irrelevant processing

## Output Schema

Each field is a structured object — NOT a plain string:
```json
{
  "name":        { "value": "JOHN SMITH", "confidence": 0.92, "source": "doctr" },
  "dateOfBirth": { "value": "01/15/1990", "confidence": 1.0,  "source": "barcode" }
}
```
`source` values: `"barcode"` | `"mrz"` | `"doctr"` | `"paddle"` | `"regex"` | `"ollama"` | `"textract"`

**server.js unwraps with**: `const fv = key => (f[key] && f[key].value) || null`
— if this breaks it means Python returned old flat schema (Docker needs rebuild).

Response also includes `warnings.glare_ratio` if > 15% of image is blown out.

## Output Fields

`name`, `dateOfBirth`, `address`, `idNumber`, `expiryDate`, `issueDate`, `sex`, `state`, `documentType`

## Key Files

| File | Purpose |
|------|---------|
| `backend/server.js` | Express API, OCR routing, Textract failsafe, `logScanResult()` |
| `backend/python-easy-ocr/app.py` | Full OCR pipeline — docTR + Ollama + PaddleOCR + PDF417 + PassportEye + regex |
| `backend/python-easy-ocr/Dockerfile` | Pre-bakes docTR + PaddleOCR models at build time |
| `backend/python-easy-ocr/requirements.txt` | Python deps: paddlepaddle, paddleocr, scikit-image, requests |
| `app/utils/ocr.ts` | `scanID(uri, side?)` — frontend HTTP client |
| `app/utils/idParser.ts` | `ParsedID` interface + client-side regex fallbacks |
| `app/scan.tsx` | Camera UI, capture/gallery, pinch-to-zoom |
| `app/form.tsx` | Verification form with editable fields |
| `app/context/ScanContext.tsx` | Cross-screen state for scanned data |

## Python Microservice Details

### Models
- **docTR**: `db_resnet50` (detection) + `parseq` (recognition)
- **PaddleOCR**: PP-OCRv5 — `use_textline_orientation=True` (PaddleOCR 3.x API)

### PaddleOCR 3.x API Breaking Changes
Old params removed — will throw `ValueError: Unknown argument`:
- `use_gpu` → removed (device auto-detected)
- `use_angle_cls` → renamed to `use_textline_orientation`
- `lang` → removed
- `show_log` → removed
- `ocr()` call: `cls=True` arg removed — use `paddle_ocr.ocr(img_array)` with no extra args

### GPU detection
Auto-detects CUDA → Apple MPS → CPU at startup. PaddleOCR has no MPS backend — falls back to CPU on Apple Silicon automatically.

### Preprocessing — docTR
Upscale to 1200px min + CLAHE on LAB luminance channel (handles glare/contrast).

### Preprocessing — PaddleOCR
1. **Perspective correction** — `cv2.getPerspectiveTransform` + `warpPerspective` on detected card corners
2. **Specular glare masking** — threshold at 240, `cv2.inpaint` INPAINT_TELEA radius=7
3. No binarization — PP-OCRv5 is neural, binarization hurts accuracy

### Spatial extraction
Label-proximity bounding-box search in normalized [0,1] coords. Returns per-field confidence (avg of value word confidences). Logs `[PY][Name]` diagnostics for all name-related fields.

### Regex fallback — NY / no-label ID format
`_extract_fields_regex_flat()` handles two layouts:
- **Standard**: label on same line or next line (FN, LN, DOB etc.)
- **NY-style** (no labels): consecutive ALL-CAPS single-word lines → LASTNAME then FIRSTNAME[,MIDDLENAME]
- **Space-split ID numbers**: joins digit-only tokens per line (e.g. `0210 049 849` → `0210049849`)
- Name separator: splits on `,` `.` or space — handles both `RISHAV,DEV` and `RISHAV.DEV`

### Ollama gap-filler
- Model: `gemma3:4b` (local, via `ollama serve`)
- URL: `host.docker.internal:11434` inside Docker, `localhost:11434` outside
- Prompt instructs model to classify dates by year, handle consecutive name lines, join split ID numbers
- Returns fields with `confidence: 0.7, source: "ollama"`
- Silently falls through if Ollama not running (`ConnectionError` caught)

### Barcode
PDF417 via pyzbar, 3-strategy decode (full → bottom-half crop → 2x upscale), AAMVA field mapping.

### MRZ
PassportEye (requires system Tesseract).

### Environment flags
- `PADDLE_ENABLED=false` — disables PaddleOCR (graceful degrade)
- `OLLAMA_ENABLED=false` — disables Ollama gap-filler
- `OLLAMA_MODEL=gemma3:4b` — Ollama model (default: gemma3:4b)
- `OLLAMA_HOST=host.docker.internal` — use `localhost` outside Docker
- `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` — skips Paddle connectivity check at startup

## Logging

All Python logs prefixed `[PY]` — distinct from `[Node]` in server.js.
Key prefixes to watch:
- `[PY] NEW REQUEST` — request banner showing which stages will run
- `[PY][Name]` — per-field name diagnostics (label found/missed, candidates tried)
- `[PY][Regex]` — what regex extracted and which path fired
- `[PY][Ollama]` — whether triggered, what it returned
- `[PY] RESULT SUMMARY` — final table with value/confidence/source per field, `<-- MISSING` markers
- `[Node] SCAN RESULT` — Node-side view of final merged fields

## server.js Notes

- `scanWithPython()` unwraps `.value` from structured fields via `fv(key)` helper
- `source` tag passed through from Python (not hardcoded in Node)
- `logScanResult()` prints `[Node]` summary after every scan
- Textract fires only if Python service throws — not on low confidence
- Dead code removed: `convertMrzDate`, `mergeData`

## Current State / Next Session

### What's working
- Full pipeline wired: docTR → regex → Ollama → PaddleOCR → Textract
- Structured output schema `{value, confidence, source}` on all fields
- NY-style no-label ID parsing (consecutive caps lines, space-split IDs)
- Ollama integration with `gemma3:4b` — graceful fallback if not running
- Logging fully labelled `[PY]` vs `[Node]`

### Blocker as of last session
Docker build failing on PaddleOCR pre-bake step. Last fix applied:
```dockerfile
RUN PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True python -c \
    "from paddleocr import PaddleOCR; PaddleOCR(use_textline_orientation=True)"
```
**Next step**: confirm Docker build completes, check `[PY]` logs appear on scan, verify NY permit fields extract correctly (name, DOB, dates, ID number).

### After Docker is stable
- Verify Ollama fills fields that regex misses on NY permit
- Test with more ID types when available
- Consider sanitizing name output (e.g. `RISHAV ACHARYA` → proper casing for form)
- Frontend form field mapping — `app/form.tsx` needs to consume new structured schema

## Environment

AWS credentials in `backend/.env` are optional — Textract is fallback only.

```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
```

## System Dependencies (macOS)

```bash
brew install tesseract zbar
pip install paddlepaddle paddleocr scikit-image requests  # local run without Docker
ollama pull gemma3:4b  # pull model once
```

## Docker

Models pre-baked at build time — no downloads at runtime:
- docTR: `db_resnet50` + `parseq`
- PaddleOCR: PP-OCRv5 (CPU, `use_textline_orientation=True`)

System deps: `tesseract-ocr`, `libgl1`, `libglib2.0-0`, `libzbar0`, `libgomp1` (PaddlePaddle OpenMP).
Build flag: `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` skips connectivity check.
