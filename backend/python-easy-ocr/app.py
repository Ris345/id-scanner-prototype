import io, os, re, base64, json
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
import torch
import requests
import pytesseract
from doctr.models import ocr_predictor
from doctr.io import DocumentFile

try:
    import zxingcpp
    ZXING_AVAILABLE = True
except Exception as e:
    ZXING_AVAILABLE = False
    print(f"[Barcode] zxing-cpp unavailable: {e}")

try:
    from pyzbar.pyzbar import decode as pyzbar_decode, ZBarSymbol
    PYZBAR_AVAILABLE = True
except Exception as e:
    PYZBAR_AVAILABLE = False
    print(f"[Barcode] pyzbar unavailable: {e}")

try:
    from passporteye import read_mrz
    PASSPORTEYE_AVAILABLE = True
except ImportError:
    PASSPORTEYE_AVAILABLE = False
    print("[MRZ] passporteye unavailable")

app = Flask(__name__)

# ── Device ─────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"[Device] {device}")

# docTR is a leftover from the pre-vision-model pipeline and is not called by /ocr. It is
# kept — and pre-baked into the image — so it can be re-wired without a rebuild, but loading
# it eagerly cost ~1GB of RSS and most of the healthcheck's start_period for nothing.
_ocr_model = None

def get_ocr_model():
    global _ocr_model
    if _ocr_model is None:
        print('[docTR] Loading model...')
        _ocr_model = ocr_predictor(det_arch='db_resnet50', reco_arch='parseq', pretrained=True).to(device)
        print('[docTR] Ready.')
    return _ocr_model

# ── Ollama ─────────────────────────────────────────────────────────────────────
OLLAMA_ENABLED = os.environ.get('OLLAMA_ENABLED', 'true').lower() != 'false'
OLLAMA_HOST    = os.environ.get('OLLAMA_HOST', 'host.docker.internal')
OLLAMA_MODEL   = os.environ.get('OLLAMA_MODEL', 'llama3.2-vision:11b')
OLLAMA_URL     = f'http://{OLLAMA_HOST}:11434/api/generate'
# Ollama evicts an idle model after ~5 min. Reloading 7.8GB takes longer than a request
# timeout of 90s allowed, so every front-side scan after a pause failed. Hold the model
# resident instead, and allow enough time for the one reload that still has to happen.
OLLAMA_KEEP_ALIVE = os.environ.get('OLLAMA_KEEP_ALIVE', '30m')
OLLAMA_TIMEOUT    = int(os.environ.get('OLLAMA_TIMEOUT', '180'))
OLLAMA_MAX_EDGE   = int(os.environ.get('OLLAMA_MAX_EDGE', '1000'))
print(f'[Ollama] {"enabled" if OLLAMA_ENABLED else "disabled"}  model={OLLAMA_MODEL}'
      f'  keep_alive={OLLAMA_KEEP_ALIVE}  timeout={OLLAMA_TIMEOUT}s')

# ── Constants ──────────────────────────────────────────────────────────────────
REQUIRED_FIELDS = ['name', 'dateOfBirth', 'idNumber']
CONF_THRESHOLD  = 0.75
VALID_OUTPUT    = {'name', 'dateOfBirth', 'idNumber', 'expiryDate', 'issueDate', 'sex', 'address', 'state'}

# documentType is deliberately NOT in VALID_OUTPUT: it is a top-level response key, not one
# of the {value, confidence, source} field objects. try_ollama() returns it separately.
DOC_TYPES = ('dl', 'state_id', 'passport')

AAMVA_MAP = {
    'DCS': 'lastName',    'DAC': 'firstName',  'DCT': 'firstName',
    'DAD': 'middleName',  'DBB': 'dateOfBirth', 'DBA': 'expiryDate',
    'DBD': 'issueDate',   'DAG': 'street',      'DAI': 'city',
    'DAJ': 'state',       'DAK': 'zip',         'DAQ': 'idNumber',
    'DBC': 'sex',         'DCA': 'vehicleClass',
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def mk(value, confidence, source):
    return {'value': value, 'confidence': confidence, 'source': source}

def is_complete(fields):
    return all(
        fields.get(f) and fields[f].get('confidence', 0) >= CONF_THRESHOLD
        for f in REQUIRED_FIELDS
    )

def merge(base, supplement):
    result = dict(base)
    for k, v in supplement.items():
        if k not in result or not result[k]:
            result[k] = v
    return result

def load_image(req):
    if 'image' in req.files:
        data = req.files['image'].read()
    elif req.is_json and req.json.get('image'):
        b64 = req.json['image']
        if ',' in b64:
            b64 = b64.split(',', 1)[1]
        data = base64.b64decode(b64)
    else:
        return None, None, 'No image provided'
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, data, None

def preprocess(img):
    h, w = img.shape[:2]
    if w < 1200:
        scale = 1200 / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

def to_png_bytes(img):
    ok, buf = cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return buf.tobytes() if ok else None

def to_jpeg_bytes(img, quality=92):
    ok, buf = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None

def downscale(img, max_edge):
    """Cap the long edge. A vision model tokenises by image area, so the upscale that
    preprocess() applies for OCR makes Ollama inference several times slower for no gain —
    card text is large and stays legible well below 1000px."""
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_edge:
        return img
    s = max_edge / longest
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

def classify_document(img):
    h, w = img.shape[:2]
    aspect = w / h
    if aspect > 1.4:
        return 'dl_or_stateid'
    elif aspect < 0.85:
        return 'passport'
    return 'unknown'


# ── Card detection ─────────────────────────────────────────────────────────────
# The app sends the full camera frame — the on-screen guide frame is decorative
# and does not crop. Locate the card ourselves so downstream stages see the card
# filling the image instead of a small region inside a tall photo.

ID1_ASPECT = 1.586  # ISO/IEC 7810 ID-1 (driver's licence / credit card)

def _order_points(pts):
    rect = np.zeros((4, 2), dtype='float32')
    s, d = pts.sum(axis=1), np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    rect[1] = pts[np.argmin(d)]  # top-right
    rect[3] = pts[np.argmax(d)]  # bottom-left
    return rect

def _four_point_transform(img, quad):
    tl, tr, br, bl = _order_points(quad)
    width  = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 2 or height < 2:
        return None
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype='float32')
    M = cv2.getPerspectiveTransform(_order_points(quad), dst)
    return cv2.warpPerspective(img, M, (width, height))

def detect_card(img):
    """Find the ID card and rectify it. Returns (image, found)."""
    h, w = img.shape[:2]
    scale = 1000.0 / max(h, w) if max(h, w) > 1000 else 1.0
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1.0 else img

    gray  = cv2.bilateralFilter(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), 11, 17, 17)
    edges = cv2.dilate(cv2.Canny(gray, 30, 200), np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    small_area = small.shape[0] * small.shape[1]

    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        if cv2.contourArea(c) < 0.05 * small_area:
            break  # sorted by area — everything after this is smaller still
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) != 4:
            continue

        card = _four_point_transform(img, approx.reshape(4, 2).astype('float32') / scale)
        if card is None:
            continue

        ch, cw = card.shape[:2]
        aspect = max(cw, ch) / min(cw, ch)
        if not (1.2 <= aspect <= 2.2):  # ID-1 is ~1.586; reject non-card rectangles
            continue
        if ch > cw:  # card held sideways — rotate to landscape
            card = cv2.rotate(card, cv2.ROTATE_90_CLOCKWISE)
            ch, cw = card.shape[:2]

        coverage = (cw * ch) / float(h * w)
        print(f'[PY][Detect] Card found {cw}x{ch}  aspect={aspect:.2f}  covers {coverage:.0%} of frame')
        return card, True

    print('[PY][Detect] No card outline found — using full frame')
    return img, False


# ── Stage 1: Barcode ───────────────────────────────────────────────────────────

def _fmt_aamva_date(s):
    return f'{s[:2]}/{s[2:4]}/{s[4:]}' if len(s) == 8 else s

def _zxing_decode(img):
    """Primary PDF417 decoder.

    ZBar is not a substitute here: its PDF417 decoder is a stub that recognises the
    symbol but returns no data, so the barcode stage silently produced nothing at all.
    Read `.bytes`, not `.text` — zxing renders control characters as a literal "<LF>"
    in `.text`, which would leave the AAMVA payload with no real line breaks.
    """
    if not ZXING_AVAILABLE:
        return None
    try:
        for r in zxingcpp.read_barcodes(img):
            if r.format == zxingcpp.BarcodeFormat.PDF417 and r.bytes:
                return bytes(r.bytes).decode('latin-1', errors='replace')
    except Exception as e:
        print(f'[PY][Barcode] zxing error: {e}')
    return None

def _pyzbar_decode(img):
    if not PYZBAR_AVAILABLE:
        return None
    hits = pyzbar_decode(img, symbols=[ZBarSymbol.PDF417])
    return hits[0].data.decode('latin-1', errors='replace') if hits else None

def _parse_aamva(raw):
    """AAMVA data elements are newline-delimited: a 3-letter code then its value.

    Splitting on a bare `[A-Z]{3}` lookahead instead truncates every text value at
    its first uppercase run ("SMITH" -> "S", "123 MAIN ST" -> "123"), so parse by line.
    """
    aamva = {}
    lines = re.split(r'[\r\n]+', raw)

    if len(lines) > 1:
        for line in lines:
            line = line.strip()
            # Exactly one element per line. Scan for its first known code rather than
            # assuming offset 0 — the header line prefixes it with the subfile
            # designators ("ANSI 636001...DL00410288...DLDAQ123456789").
            # Taking the rest of the line as the value is also what keeps a value that
            # merely contains a code ("DCS" + "DACOSTA") from being split at it.
            for i in range(len(line) - 2):
                code = line[i:i + 3]
                if code in AAMVA_MAP:
                    val = line[i + 3:].strip()
                    if val:
                        aamva.setdefault(AAMVA_MAP[code], val)
                    break
    else:
        # Malformed payload with no line breaks — split on the known code set only,
        # never on any three capitals.
        codes = '|'.join(AAMVA_MAP)
        for m in re.finditer(rf'({codes})(.*?)(?=(?:{codes})|\Z)', raw, re.DOTALL):
            val = m.group(2).strip()
            if val:
                aamva.setdefault(AAMVA_MAP[m.group(1)], val)
    return aamva


def try_barcode(img):
    if not (ZXING_AVAILABLE or PYZBAR_AVAILABLE):
        return {}, 'unknown'
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    raw = (
        # zxing copes with perspective and full frames, so give it the whole image first.
        _zxing_decode(gray) or
        _zxing_decode(gray[h // 2:, :]) or
        _zxing_decode(cv2.resize(gray, (w * 2, h * 2))) or
        _pyzbar_decode(img) or
        _pyzbar_decode(img[h // 2:, :]) or
        _pyzbar_decode(cv2.resize(img, (w * 2, h * 2)))
    )
    if not raw:
        print('[PY][Barcode] No PDF417 found')
        return {}, 'unknown'

    print('[PY][Barcode] Found PDF417, parsing AAMVA...')
    aamva = _parse_aamva(raw)

    if not aamva:
        return {}, 'unknown'

    name = ' '.join(filter(None, [aamva.get('firstName', ''), aamva.get('middleName', ''), aamva.get('lastName', '')])) or None
    street, city, state, zip_ = aamva.get('street',''), aamva.get('city',''), aamva.get('state',''), aamva.get('zip','')
    addr = ', '.join(filter(None, [street, city, state]))
    if zip_: addr = f'{addr} {zip_}'.strip() if addr else zip_
    sex = {'1': 'M', '2': 'F', '9': 'X'}.get(aamva.get('sex', ''), aamva.get('sex') or None)

    fields = {}
    if name:                   fields['name']        = mk(name, 1.0, 'barcode')
    if aamva.get('dateOfBirth'): fields['dateOfBirth'] = mk(_fmt_aamva_date(aamva['dateOfBirth']), 1.0, 'barcode')
    if addr.strip():           fields['address']     = mk(addr.strip(), 1.0, 'barcode')
    if aamva.get('idNumber'):  fields['idNumber']    = mk(aamva['idNumber'], 1.0, 'barcode')
    if aamva.get('expiryDate'): fields['expiryDate'] = mk(_fmt_aamva_date(aamva['expiryDate']), 1.0, 'barcode')
    if aamva.get('issueDate'): fields['issueDate']   = mk(_fmt_aamva_date(aamva['issueDate']), 1.0, 'barcode')
    if sex:                    fields['sex']         = mk(sex, 1.0, 'barcode')
    if state:                  fields['state']       = mk(state, 1.0, 'barcode')

    doc_type = 'dl' if aamva.get('vehicleClass') else 'state_id'
    print(f'[PY][Barcode] {len(fields)} fields  doc_type={doc_type}')
    return fields, doc_type


# ── Stage 2: MRZ ──────────────────────────────────────────────────────────────

def _mrz_date(s):
    if len(s) == 6:
        yy, mm, dd = s[:2], s[2:4], s[4:]
        year = f'19{yy}' if int(yy) > 30 else f'20{yy}'
        return f'{mm}/{dd}/{year}'
    return s

def try_mrz(img_bytes):
    if not PASSPORTEYE_AVAILABLE:
        return {}, 'unknown'
    try:
        mrz = read_mrz(io.BytesIO(img_bytes))
        if not mrz:
            print('[PY][MRZ] No MRZ found')
            return {}, 'unknown'
        data = mrz.to_dict()
        print(f'[PY][MRZ] Found  valid_score={data.get("valid_score")}')

        surname = data.get('surname', '')
        names   = data.get('names', '')
        name    = f'{names} {surname}'.strip() if names else surname
        conf    = 0.98 if (data.get('valid_score') or 0) >= 80 else 0.50

        fields = {}
        if name:                  fields['name']        = mk(name, conf, 'mrz')
        if data.get('date_of_birth'): fields['dateOfBirth'] = mk(_mrz_date(data['date_of_birth']), conf, 'mrz')
        if data.get('number'):    fields['idNumber']    = mk(data['number'], conf, 'mrz')
        if data.get('expiry_date'): fields['expiryDate'] = mk(_mrz_date(data['expiry_date']), conf, 'mrz')
        if data.get('sex'):       fields['sex']         = mk(data['sex'], conf, 'mrz')
        if data.get('country'):   fields['state']       = mk(data['country'], conf, 'mrz')

        t = data.get('type', '')
        doc_type = 'passport' if t.startswith('P') else 'id_card' if t[:1] in ('I','A','C') else 'unknown'
        return fields, doc_type
    except Exception as e:
        print(f'[PY][MRZ] Error: {e}')
        return {}, 'unknown'


# ── Stage 4: Tesseract — raw text safety net ───────────────────────────────────
# Runs only when the structured stages came back incomplete. It does not produce
# fields; it produces the `raw_text` the form screen shows so that a scan which
# barcode/MRZ/Ollama all missed still hands the user something to type from,
# instead of "No text detected".

def try_tesseract(img):
    try:
        # Tesseract works best on high-contrast grayscale — convert and threshold
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        pil_img = Image.fromarray(thresh)

        # PSM 11 = sparse text, handles scattered labels on ID cards
        raw = pytesseract.image_to_string(pil_img, config='--psm 11 --oem 3')
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        raw_clean = '\n'.join(lines)
        print(f'[PY][Tesseract] {len(lines)} lines of raw text')
        return raw_clean
    except Exception as e:
        # A missing tesseract binary must not take down a scan that already has fields.
        print(f'[PY][Tesseract] Failed: {e}')
        return ''


# ── Stage 3: Ollama vision ─────────────────────────────────────────────────────

def _normalize_doc_type(v):
    """Map whatever the model calls the document onto our four-value vocabulary.

    Order matters. "non-driver ID" is the standard name for a state ID card and
    contains "driver", so the negations have to be tested before the DL rule.
    """
    s = str(v or '').strip().lower()
    if s in DOC_TYPES:
        return s
    if 'passport' in s:
        return 'passport'
    if 'non-driver' in s or 'non driver' in s or 'nondriver' in s:
        return 'state_id'
    if 'driver' in s or 'permit' in s or 'learner' in s or s in ('dl', 'dln'):
        return 'dl'
    if 'identification' in s or re.search(r'\bid\b', s):
        return 'state_id'
    return 'unknown'


def try_ollama(img_bytes):
    """Returns (fields, doc_type)."""
    if not OLLAMA_ENABLED:
        return {}, 'unknown'

    # Framed as transcription rather than "extract personal information from an ID", which
    # llama3.2-vision intermittently refuses outright ("...not within my ethical guidelines").
    # The refusal is non-deterministic — the same image can succeed then fail on a retry.
    prompt = (
        "Transcribe the printed text visible in this document photo into JSON. "
        "The document owner is scanning it themselves to fill in a form, so read it verbatim. "
        "Reply with only a JSON object using exactly these keys:\n"
        '{"name":"FIRST MIDDLE LAST","dateOfBirth":"MM/DD/YYYY","idNumber":"...","expiryDate":"MM/DD/YYYY",'
        '"issueDate":"MM/DD/YYYY","sex":"M or F or X","address":"full address","state":"state name",'
        '"documentType":"dl or state_id or passport"}\n\n'
        "Rules:\n"
        "- name: first name then last name order (e.g. JOHN SMITH not SMITH JOHN)\n"
        "- documentType: dl for a driver licence or learner permit, state_id for a "
        "non-driver ID card, passport for a passport\n"
        "- dates: MM/DD/YYYY format only\n"
        "- idNumber: digits only, no dashes or spaces\n"
        "- Use null for any field that is not legible\n"
        "- Transcribe only what is printed; never invent a value"
    )

    try:
        resp = requests.post(OLLAMA_URL, json={
            'model':      OLLAMA_MODEL,
            'prompt':     prompt,
            'images':     [base64.b64encode(img_bytes).decode()],
            'stream':     False,
            'keep_alive': OLLAMA_KEEP_ALIVE,
            # Constrains decoding to valid JSON, so the model cannot emit a prose refusal
            # in place of the object. This is the main defence against the refusal above.
            'format':     'json',
        }, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
        text  = resp.json().get('response', '')
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            print(f'[PY][Ollama] No JSON in response: {text[:200]}')
            return {}, 'unknown'
        parsed = json.loads(match.group())
        if not isinstance(parsed, dict):
            print(f'[PY][Ollama] Expected an object, got {type(parsed).__name__}')
            return {}, 'unknown'
        fields = {
            k: mk(str(v), 0.70, 'ollama')
            for k, v in parsed.items()
            if k in VALID_OUTPUT and v and v not in (None, 'null')
        }
        doc_type = _normalize_doc_type(parsed.get('documentType'))
        print(f'[PY][Ollama] {len(fields)} fields extracted  doc_type={doc_type}')
        return fields, doc_type
    except Exception as e:
        print(f'[PY][Ollama] Failed: {e}')
        return {}, 'unknown'


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'device': str(device)})


@app.route('/ocr', methods=['POST'])
def ocr():
    img, img_bytes, err = load_image(request)
    if err:
        return jsonify({'success': False, 'error': err}), 400

    body     = request.json if request.is_json else {}
    side     = body.get('side')
    fields   = {}
    doc_type = 'unknown'
    source   = 'none'

    # Crop to the card before anything else — the client sends the whole camera frame.
    card, card_found = detect_card(img)
    card = preprocess(card)
    card_bytes = to_png_bytes(card) or img_bytes

    # Only trust the aspect-ratio classifier once we have an actual card crop;
    # on a full portrait photo it reports 'passport' and wrongly skips barcode.
    doc_class = classify_document(card) if card_found else 'unknown'
    print(f'[PY] NEW REQUEST  side={side}  doc_class={doc_class}  card_found={card_found}')

    # Stage 1: Barcode — try the crop, then fall back to the untouched frame,
    # since CLAHE and the warp can both disturb a PDF417.
    if side != 'front' and doc_class in ('dl_or_stateid', 'unknown'):
        barcode_fields, barcode_type = try_barcode(card)
        if not barcode_fields and card_found:
            barcode_fields, barcode_type = try_barcode(img)
        if barcode_fields:
            fields   = merge(fields, barcode_fields)
            doc_type = barcode_type
            if is_complete(fields):
                print('[PY] Early exit: barcode')
                return _respond(fields, 'barcode', doc_type, card)

    # Stage 2: MRZ
    if side != 'back' and doc_class in ('passport', 'unknown'):
        mrz_fields, mrz_type = try_mrz(card_bytes)
        if mrz_fields:
            fields   = merge(fields, mrz_fields)
            doc_type = mrz_type if mrz_type != 'unknown' else doc_type
            if is_complete(fields):
                print('[PY] Early exit: MRZ')
                return _respond(fields, 'mrz', doc_type, card)

    # Stage 3: Ollama vision — gets its own smaller copy; see downscale().
    ollama_bytes = to_jpeg_bytes(downscale(card, OLLAMA_MAX_EDGE)) or card_bytes
    ollama_fields, ollama_type = try_ollama(ollama_bytes)
    if ollama_fields:
        fields = merge(fields, ollama_fields)
        source = 'llama-vision'
    elif fields:
        source = 'barcode' if doc_type in ('dl', 'state_id') else 'mrz'
    # Barcode and MRZ read the document's own machine-readable declaration, so they win.
    # The model only fills the gap they leave — chiefly a front-of-card scan.
    if doc_type == 'unknown' and ollama_type != 'unknown':
        doc_type = ollama_type

    # Stage 4: Tesseract safety net. Only worth its ~1-2s when the structured stages
    # left us short — a complete result has nothing to gain from raw text.
    raw_text = ''
    if not is_complete(fields):
        raw_text = try_tesseract(card)
        if raw_text and not fields:
            # Nothing structured at all. Saying so lets the client show the raw text
            # rather than the "No text detected" dead end.
            source = 'tesseract'

    return _respond(fields, source, doc_type, card, raw_text)


def _respond(fields, source, doc_type, img, raw_text=''):
    gray        = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    glare_ratio = float(np.count_nonzero(gray > 240)) / gray.size
    warnings    = ['glare_detected'] if glare_ratio > 0.15 else []
    confs       = [fields[f]['confidence'] for f in REQUIRED_FIELDS if fields.get(f)]
    confidence  = round(min(confs), 4) if confs else 0.0

    _log(fields, source, confidence, doc_type, raw_text)
    return jsonify({
        'success':      True,
        'source':       source,
        'documentType': doc_type,
        'confidence':   confidence,
        'fields':       fields,
        'warnings':     warnings,
        # server.js reads this as `raw_text` and forwards it to the client as `rawText`.
        'raw_text':     raw_text,
    })


def _log(fields, source, confidence, doc_type='unknown', raw_text=''):
    SEP = '─' * 50
    print(f'\n[PY] {"═"*50}')
    print(f'[PY] RESULT  source={source}  conf={confidence:.2f}  doc_type={doc_type}')
    print(f'[PY] {SEP}')
    for f in ['name', 'dateOfBirth', 'idNumber', 'expiryDate', 'issueDate', 'sex', 'address', 'state']:
        entry = fields.get(f)
        if entry:
            print(f'[PY]   {f:<14} conf={entry["confidence"]:.2f}  src={entry["source"]}')
        else:
            print(f'[PY]   {f:<14} <-- MISSING')
    if raw_text:
        print(f'[PY] {SEP}')
        print(f'[PY] raw_text ({len(raw_text)} chars):')
        for line in raw_text.splitlines():
            print(f'[PY]   {line}')
    print(f'[PY] {"═"*50}\n')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3002, debug=False)
