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

print("[docTR] Loading model...")
ocr_model = ocr_predictor(det_arch='db_resnet50', reco_arch='parseq', pretrained=True).to(device)
print("[docTR] Ready.")

# ── Ollama ─────────────────────────────────────────────────────────────────────
OLLAMA_ENABLED = os.environ.get('OLLAMA_ENABLED', 'true').lower() != 'false'
OLLAMA_HOST    = os.environ.get('OLLAMA_HOST', 'host.docker.internal')
OLLAMA_MODEL   = os.environ.get('OLLAMA_MODEL', 'llama3.2-vision:11b')
OLLAMA_URL     = f'http://{OLLAMA_HOST}:11434/api/generate'
print(f'[Ollama] {"enabled" if OLLAMA_ENABLED else "disabled"}  model={OLLAMA_MODEL}')

# ── Constants ──────────────────────────────────────────────────────────────────
REQUIRED_FIELDS = ['name', 'dateOfBirth', 'idNumber']
CONF_THRESHOLD  = 0.75
VALID_OUTPUT    = {'name', 'dateOfBirth', 'idNumber', 'expiryDate', 'issueDate', 'sex', 'address', 'state'}

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

def classify_document(img):
    h, w = img.shape[:2]
    aspect = w / h
    if aspect > 1.4:
        return 'dl_or_stateid'
    elif aspect < 0.85:
        return 'passport'
    return 'unknown'


# ── Stage 1: Barcode ───────────────────────────────────────────────────────────

def _fmt_aamva_date(s):
    return f'{s[:2]}/{s[2:4]}/{s[4:]}' if len(s) == 8 else s

def _pyzbar_decode(img):
    hits = pyzbar_decode(img, symbols=[ZBarSymbol.PDF417])
    return hits[0].data.decode('latin-1', errors='replace') if hits else None

def try_barcode(img):
    if not PYZBAR_AVAILABLE:
        return {}, 'unknown'
    h, w = img.shape[:2]
    raw = (
        _pyzbar_decode(img) or
        _pyzbar_decode(img[h // 2:, :]) or
        _pyzbar_decode(cv2.resize(img, (w * 2, h * 2)))
    )
    if not raw:
        print('[PY][Barcode] No PDF417 found')
        return {}, 'unknown'

    print('[PY][Barcode] Found PDF417, parsing AAMVA...')
    aamva = {}
    for m in re.finditer(r'([A-Z]{3})(.+?)(?=[A-Z]{3}|\Z)', raw, re.DOTALL):
        code, val = m.group(1), m.group(2).strip()
        if code in AAMVA_MAP:
            aamva[AAMVA_MAP[code]] = val

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


# ── Stage 3: Tesseract — raw text extraction ───────────────────────────────────

def try_tesseract(img):
    # Tesseract works best on high-contrast grayscale — convert and threshold
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pil_img = Image.fromarray(thresh)

    # PSM 11 = sparse text, handles scattered labels on ID cards
    raw = pytesseract.image_to_string(pil_img, config='--psm 11 --oem 3')
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    raw_clean = '\n'.join(lines)
    print(f'[PY][Tesseract] {len(lines)} lines')
    print(f'[PY][Tesseract] Raw text:\n{raw_clean}')
    return raw_clean


# ── Stage 3: Ollama vision ─────────────────────────────────────────────────────

def try_ollama(img_bytes, partial_fields):
    if not OLLAMA_ENABLED:
        return {}

    prompt = (
        "This is a photo of a government ID card (driver's license, permit, or passport). "
        "Extract the fields and return ONLY valid JSON, no explanation:\n"
        '{"name":"FIRST MIDDLE LAST","dateOfBirth":"MM/DD/YYYY","idNumber":"...","expiryDate":"MM/DD/YYYY",'
        '"issueDate":"MM/DD/YYYY","sex":"M or F or X","address":"full address","state":"state name"}\n\n'
        "Rules:\n"
        "- name: first name then last name order (e.g. JOHN SMITH not SMITH JOHN)\n"
        "- dates: MM/DD/YYYY format only\n"
        "- idNumber: digits only, no dashes or spaces\n"
        "- Use null for fields you cannot read\n"
        "- Do not guess"
    )

    try:
        resp = requests.post(OLLAMA_URL, json={
            'model':  OLLAMA_MODEL,
            'prompt': prompt,
            'images': [base64.b64encode(img_bytes).decode()],
            'stream': False,
        }, timeout=90)
        resp.raise_for_status()
        text  = resp.json().get('response', '')
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            print(f'[PY][Ollama] No JSON in response: {text[:200]}')
            return {}
        parsed = json.loads(match.group())
        fields = {
            k: mk(str(v), 0.70, 'ollama')
            for k, v in parsed.items()
            if k in VALID_OUTPUT and v and v not in (None, 'null')
        }
        print(f'[PY][Ollama] {len(fields)} fields extracted')
        return fields
    except Exception as e:
        print(f'[PY][Ollama] Failed: {e}')
        return {}


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
    source   = 'doctr'

    doc_class = classify_document(img)
    print(f'[PY] NEW REQUEST  side={side}  doc_class={doc_class}')

    # Stage 1: Barcode
    if side != 'front' and doc_class in ('dl_or_stateid', 'unknown'):
        barcode_fields, barcode_type = try_barcode(img)
        if barcode_fields:
            fields   = merge(fields, barcode_fields)
            doc_type = barcode_type
            if is_complete(fields):
                print('[PY] Early exit: barcode')
                return _respond(fields, 'barcode', doc_type, img)

    # Stage 2: MRZ
    if side != 'back' and doc_class in ('passport', 'unknown'):
        mrz_fields, mrz_type = try_mrz(img_bytes)
        if mrz_fields:
            fields   = merge(fields, mrz_fields)
            doc_type = mrz_type if mrz_type != 'unknown' else doc_type
            if is_complete(fields):
                print('[PY] Early exit: MRZ')
                return _respond(fields, 'mrz', doc_type, img)

    # Stage 3: Ollama vision
    ollama_fields = try_ollama(img_bytes, fields)
    if ollama_fields:
        fields = merge(fields, ollama_fields)
        source = 'llama-vision'

    return _respond(fields, source, doc_type, img)


def _respond(fields, source, doc_type, img):
    gray        = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    glare_ratio = float(np.count_nonzero(gray > 240)) / gray.size
    warnings    = ['glare_detected'] if glare_ratio > 0.15 else []
    confs       = [fields[f]['confidence'] for f in REQUIRED_FIELDS if fields.get(f)]
    confidence  = round(min(confs), 4) if confs else 0.0

    _log(fields, source, confidence)
    return jsonify({
        'success':      True,
        'source':       source,
        'documentType': doc_type,
        'confidence':   confidence,
        'fields':       fields,
        'warnings':     warnings,
    })


def _log(fields, source, confidence):
    SEP = '─' * 50
    print(f'\n[PY] {"═"*50}')
    print(f'[PY] RESULT  source={source}  conf={confidence:.2f}')
    print(f'[PY] {SEP}')
    for f in ['name', 'dateOfBirth', 'idNumber', 'expiryDate', 'issueDate', 'sex', 'address', 'state']:
        entry = fields.get(f)
        if entry:
            print(f'[PY]   {f:<14} conf={entry["confidence"]:.2f}  src={entry["source"]}')
        else:
            print(f'[PY]   {f:<14} <-- MISSING')
    print(f'[PY] {"═"*50}\n')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3002, debug=False)
