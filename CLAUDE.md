# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# ID Scanner Prototype

Government ID document scanning app that extracts structured data from photos of driver's licenses, passports, and national IDs.

## Architecture

Three-tier system:
- **Frontend**: Expo (React Native) — `app/` — runs on the host
- **Backend**: Node.js/Express on port 3001 — `backend/server.js` — Docker
- **Python OCR microservice**: Flask on port 3002 — `backend/python-easy-ocr/app.py` — Docker

Inside the compose network the Node service reaches Python at `http://python-ocr:3002`
(`PYTHON_OCR_URL`), not `localhost` — localhost in a container is the container itself.
Both ports are still published, so host tools and the Expo app keep using `localhost`.

## Dev Commands

Both backend services run in Docker. `docker-compose.yml` lives in `backend/`, not
`backend/python-easy-ocr/`.

```bash
# Whole backend stack (Node :3001 + Python :3002). Node waits for Python's healthcheck.
cd backend && docker-compose up -d
docker-compose logs -f            # tail both
docker-compose logs -f python-ocr # tail just the [PY] pipeline

# Ollama — runs on the HOST, not in Docker (see below). Separate terminal, before scanning.
OLLAMA_HOST=0.0.0.0 ollama serve  # Metal GPU on Mac; check port: lsof -i :11434

# Frontend (host)
npx expo start
```

Containers are `id-scanner-backend-1` and `id-scanner-python-ocr-1`.

### Applying changes

```bash
# server.js is bind-mounted read-only — just restart, no rebuild
cd backend && docker-compose restart backend

# app.py is baked into the image; copy it in for a fast iteration
docker cp python-easy-ocr/app.py id-scanner-python-ocr-1:/app/app.py && docker restart id-scanner-python-ocr-1

# Rebuild (needed when requirements.txt, package.json, or a Dockerfile changes)
docker-compose build && docker-compose up -d --force-recreate
```

### Why Ollama is not in the compose stack

A container cannot reach Apple's Metal GPU, so an 11B vision model inside Docker runs on
CPU — minutes per scan instead of ~30–60s. `python-ocr` therefore talks to the host's
Ollama via `host.docker.internal` (the `extra_hosts: host-gateway` line makes that work on
Linux too). To containerise it anyway — sensible on a Linux/CUDA box, or for full isolation:

```bash
docker-compose --profile ollama up -d
OLLAMA_HOST=ollama docker-compose up -d python-ocr   # point the service at it
docker exec id-scanner-ollama-1 ollama pull llama3.2-vision:11b
```

### Compose project name

`name: id-scanner` is pinned at the top of `docker-compose.yml`. Without it Compose derives
the project name from the directory (`backend`) and will **adopt and recreate identically
named services from any other project on the machine that also has a `backend/` dir**.
Do not remove it, and never run `docker-compose down --remove-orphans` from this directory.

## OCR Pipeline

**Priority order**: barcode > MRZ > llama3.2-vision > Tesseract raw text > Textract (crash fallback only)

```
Image (full camera frame)
  ↓
0. detect_card()               → locate card outline, perspective-correct, crop
   preprocess()                → upscale to 1200px + CLAHE
  ↓
1. PDF417 barcode decode       → if found + parsed → return (confidence 1.0)
  ↓
2. MRZ (PassportEye)           → if found + checksum passes → return (confidence 0.98)
  ↓
3. llama3.2-vision:11b (Ollama) → sends image directly, returns structured JSON fields
  ↓
4. Tesseract (raw text only)   → only if is_complete() is still false; fills `raw_text`,
                                 never fields. Source becomes "tesseract" if it is the
                                 only thing that found anything.
```

- `POST /api/scan` → Python microservice always first
- **Ollama receives the raw image bytes** — vision model reads the ID card directly, no OCR pre-step
- **Textract**: only fires in Node.js if the Python service is *unreachable* — never on a rejected image (see "The Textract boundary")
- Optional `side: 'front' | 'back'` in request body skips irrelevant stages
- Document classifier (aspect ratio) gates which stages run: `dl_or_stateid` | `passport` | `unknown`
- **The classifier only runs on a successful card crop.** On a full portrait frame it is meaningless — it classifies as `passport` and would skip the barcode stage on a driver's license. When `detect_card()` finds nothing, doc_class is forced to `unknown` so every stage still runs.
- Node.js timeout: 120s (llama3.2-vision:11b is slow on first load, ~30–60s)

### documentType

Set by whichever stage identified the document: barcode (`dl` if a `DCA` vehicle class is
present, else `state_id`), MRZ (from the MRZ type character), or — when both left it
`unknown`, which is the normal case for a **front**-of-card scan — the vision model.

`documentType` is deliberately **not** in `VALID_OUTPUT`. It is a top-level response key,
not one of the `{value, confidence, source}` field objects, so `try_ollama()` returns it as
the second element of its tuple. Adding it to `VALID_OUTPUT` would bury it inside `fields`,
where `server.js` (which reads `ocrResult.documentType`) would never see it.

`_normalize_doc_type()` maps the model's free text onto `dl` | `state_id` | `passport` |
`unknown`. **Order of its checks matters**: "non-driver ID" is the standard name for a state
ID card and contains "driver", so the negations are tested before the DL rule.

### Unused code in pipeline

- `ocr_model` (docTR `db_resnet50` + `parseq`) is still **not called** in the `/ocr` route, but is now behind `get_ocr_model()` and loads lazily — eagerly loading it cost ~1GB RSS and most of the healthcheck's `start_period` for nothing. The Dockerfile still pre-bakes it so re-wiring needs no rebuild.
- `parseIDText()` in `app/utils/idParser.ts` is legacy regex parsing — **not called** from `scan.tsx` or `ocr.ts`; the file's live purpose is the `ParsedID` interface

### Client-side cropping

`cropToGuideFrame()` in `scan.tsx` maps the on-screen guide frame onto the captured photo
and crops to it (+12% padding, so a loosely framed card isn't clipped — the PDF417 sits at
the card edge). Both the container and the guide `View` are measured with `measureInWindow`;
the preview fills the view "cover"-style, so the mapping divides out that scale and centring.

It returns the **uncropped** URI whenever the geometry can't be trusted — unmeasured views,
a landscape photo buffer behind a portrait preview, or a degenerate rect. A wrong crop is
worse than none, since `detect_card()` can still find the card server-side.

Camera captures only. **Gallery picks are never cropped** — there's no guide frame to map.

## Output Schema

Each field is a structured object — NOT a plain string:
```json
{
  "name":        { "value": "JOHN SMITH", "confidence": 0.92, "source": "ollama" },
  "dateOfBirth": { "value": "01/15/1990", "confidence": 1.0,  "source": "barcode" }
}
```

Per-field `source` values: `"barcode"` | `"mrz"` | `"ollama"` | (Textract fields are plain strings, no `.source`)

Top-level response `source` field uses: `"barcode"` | `"mrz"` | `"llama-vision"` | `"tesseract"` | `"textract"` | `"none"`

**server.js unwraps with**: `const fv = key => (f[key] && f[key].value) || null`

Response also includes `warnings: ["glare_detected"]` if > 15% of image is blown out.

### raw_text

Python returns `raw_text` (snake_case); `server.js` forwards it to the client as `rawText`;
`app/form.tsx` renders it and `scan.tsx` uses `hasRawText` (>20 chars) to decide whether a
scan counts as a failure. That chain was dead for several iterations — Python had stopped
emitting `raw_text` entirely, so `hasRawText` was permanently false and any scan that
produced no fields hit "No text detected", even when the card was perfectly legible.

`raw_text` is `''` on any complete result. Tesseract only runs when `is_complete()` is
false, so the ~1–2s it costs is never paid on a clean barcode or MRZ scan.

**A vision-only scan always pays it**, though: Ollama emits fields at `0.70` and
`CONF_THRESHOLD` is `0.75`, so `is_complete()` is false by construction on a llama-vision
result. That is deliberate rather than accidental — a 0.70 read is exactly the case where
the user checking the form wants the verbatim text next to the parsed fields. Raising
Ollama's confidence above 0.75 would silently switch this off.

## Output Fields

`name`, `dateOfBirth`, `address`, `idNumber`, `expiryDate`, `issueDate`, `sex`, `state`, `documentType`

## Key Files

| File | Purpose |
|------|---------|
| `backend/server.js` | Express API, Python call, Textract crash fallback, `logScanResult()` |
| `backend/python-easy-ocr/app.py` | OCR pipeline — barcode + MRZ + Ollama vision + Tesseract raw text |
| `backend/python-easy-ocr/Dockerfile` | Pre-bakes docTR models at build time (loaded lazily, not used in scan path) |
| `backend/python-easy-ocr/requirements.txt` | Python deps |
| `app/utils/ocr.ts` | `scanID(uri, side?)` — frontend HTTP client |
| `app/utils/idParser.ts` | `ParsedID` interface (+ legacy `parseIDText` not used in scan flow) |
| `app/scan.tsx` | Camera UI, capture/gallery, pinch-to-zoom, `cropToGuideFrame()` |
| `app/form.tsx` | Verification form with editable fields |
| `app/context/ScanContext.tsx` | Cross-screen state for scanned data |

## Python Microservice Details

### Models
- **docTR**: `db_resnet50` (detection) + `parseq` (recognition) — **lazy** behind `get_ocr_model()`, still not called in the scan route
- **Tesseract**: `try_tesseract()` is stage 4 — raw text only, never fields. Wrapped in a `try` so a missing `tesseract` binary degrades to `''` instead of failing a scan that already has fields.

### GPU detection
Auto-detects CUDA → Apple MPS → CPU at startup.

### Card detection
`detect_card()` — Canny + contour search for the largest 4-sided shape covering >5% of the
frame with an aspect between 1.2 and 2.2 (ID-1 is ~1.586), then `getPerspectiveTransform`
to rectify it. Sideways cards are rotated to landscape. Falls back to the full frame when
no card outline is found.

This runs on **every** image regardless of client-side cropping — it is what corrects
perspective skew, and gallery uploads are never cropped by the client.

### Preprocessing
Upscale to 1200px min + CLAHE on LAB luminance channel. Applied to the card crop, before all stages.
Numpy array converted to PNG bytes via `to_png_bytes()` for the MRZ and Ollama stages.

### Barcode
PDF417 via **zxing-cpp**, then pyzbar as a secondary attempt. Each gets the same 3 strategies
(full image → bottom-half crop → 2x upscale). Tried on the card crop first, then on the
untouched frame — the warp and CLAHE can both disturb a PDF417.

**Do not rely on pyzbar/ZBar for PDF417.** Its PDF417 decoder is a stub: it recognises the
symbol but returns no data, even on a clean, high-resolution, machine-generated barcode.
While it was the only decoder, the barcode stage never produced a single field.

`_zxing_decode()` reads `r.bytes`, **not** `r.text` — zxing renders control characters as a
literal `"<LF>"` in `.text`, which leaves the AAMVA payload with zero real line breaks and
sends `_parse_aamva()` down its malformed-input branch.

`_parse_aamva()` parses **one element per line** (AAMVA elements are newline-delimited).
Do not split on a bare `[A-Z]{3}` lookahead: that truncates every text value at its first
uppercase run (`SMITH` → `S`, `123 MAIN ST` → `123`) and returns the wreckage at confidence 1.0.
It scans each line for its first known code rather than assuming offset 0, since the header
line prefixes it with subfile designators (`ANSI 636001...DLDAQ123456789`).

### MRZ
PassportEye (requires system Tesseract). Confidence 0.98 if checksum passes, 0.50 if not.

### Ollama vision
- Model: `llama3.2-vision:11b` (local, via `ollama serve`)
- URL: `host.docker.internal:11434` inside Docker, `localhost:11434` outside
- **Receives the raw image bytes** — vision model reads the ID card directly
- Prompt instructs model to return structured JSON with all ID fields
- Returns fields with `confidence: 0.70, source: "ollama"`
- Silently falls through if Ollama not running
- Pre-warm before first scan: `ollama run llama3.2-vision:11b "hi"`

### Environment flags
- `OLLAMA_ENABLED=false` — disables Ollama structuring
- `OLLAMA_MODEL=<model>` — override model (default: `llama3.2-vision:11b`)
- `OLLAMA_HOST=host.docker.internal` — use `localhost` outside Docker

## Logging

All Python logs prefixed `[PY]` — distinct from `[Node]` in server.js.
Key prefixes to watch:
- `[PY] NEW REQUEST` — request banner with side and doc_class
- `[PY][Barcode]` — PDF417 decode result
- `[PY][MRZ]` — MRZ parse result
- `[PY][Ollama]` — field count returned
- `[PY] RESULT` — final table with confidence/source per field, `<-- MISSING` markers
- `[Node] SCAN RESULT` — Node-side view of final merged fields

## server.js Notes

- `scanWithPython()` unwraps `.value` from structured fields via `fv(key)` helper
- `confidence` passed through from Python response
- `logScanResult()` prints `[Node]` summary after every scan
- **Textract fires only if the Python service is unreachable** — not on low confidence, and not on a rejected image

### The Textract boundary

Textract is the one path that sends the user's ID off the machine, so what may cross it is
narrow and deliberate: **only a genuinely unreachable Python service.**

- Python returns **4xx** for a bad *input* (undecodable bytes, empty body, no image). `scanWithPython()` tags these `err.badInput` and `/api/scan` returns 400 **without** calling Textract — no cloud OCR can turn an undecodable upload into fields, so the call would only leak the image to reach the same answer.
- Python's `@app.errorhandler(Exception)` guarantees a **JSON** reply. Flask's default HTML error page made Node's `res.json()` throw, which is indistinguishable from the service being down — so any unhandled bug used to route the image to AWS. It re-raises `HTTPException` with its own status, or a 404/405 would report as a 500 and hit that same branch.
- `cv2.imdecode()` returns `None` instead of raising on a corrupt image; `load_image()` checks for it explicitly. Without that check `cvtColor` raised, which was exactly the HTML-500 case above.
- `backend/package.json` contains legacy deps (`openai`, `@google-cloud/vision`, `tesseract.js`, `sharp`, `mrz`) from prior pipeline iterations — they are not imported in `server.js`

## Environment

AWS credentials are optional — Textract is crash fallback only. They live in `.env` at the
**project root**, not `backend/.env`: `server.js` loads `path.join(__dirname, '../.env')`.
(In Docker `__dirname` is `/app`, so that path misses — compose passes the AWS vars through
`environment:` instead.)

```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
```

## System Dependencies (macOS)

```bash
brew install tesseract zbar
ollama pull llama3.2-vision:11b  # pull model once (~8GB)
```

## Docker

Models pre-baked at build time — no downloads at runtime:
- docTR: `db_resnet50` + `parseq`

System deps: `tesseract-ocr`, `libgl1`, `libglib2.0-0`, `libzbar0`.

`PYTHONUNBUFFERED=1` is set in `docker-compose.yml`. Without it the `[PY]` pipeline logs
sit in stdout's buffer and never reach `docker-compose logs` while you're debugging a scan.
`zxing-cpp` ships prebuilt wheels, so it needs no extra system packages.
