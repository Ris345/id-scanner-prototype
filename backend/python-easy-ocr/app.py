import io
import re
import base64
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

try:
    from passporteye import read_mrz
    PASSPORTEYE_AVAILABLE = True
except ImportError:
    PASSPORTEYE_AVAILABLE = False
    print("PassportEye not available — install passporteye + tesseract")

try:
    from pyzbar.pyzbar import decode as pyzbar_decode, ZBarSymbol
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False
    print("pyzbar not available — barcode decoding disabled")

import torch
from doctr.models import ocr_predictor
from doctr.io import DocumentFile

app = Flask(__name__)

# Auto-detect best available compute device and log it
if torch.cuda.is_available():
    print("[Device] CUDA GPU detected")
elif torch.backends.mps.is_available():
    print("[Device] Apple MPS (Metal) detected")
else:
    print("[Device] No GPU detected — running on CPU")

print("Loading docTR model (db_resnet50 + parseq)...")
model = ocr_predictor(det_arch='db_resnet50', reco_arch='parseq', pretrained=True)
print("docTR ready.")

# ── Spatial extraction constants ───────────────────────────────────────────────
SAME_ROW_Y_TOLERANCE = 0.03   # ~3% of page height, absorbs OCR vertical jitter
SAME_COL_X_TOLERANCE = 0.05   # for below-label search

LABEL_ALIASES = {
    'lastName':    ['LN', 'LAST', 'LAST NAME', 'SURNAME'],
    'firstName':   ['FN', 'FIRST', 'FIRST NAME', 'GIVEN', 'GIVEN NAME'],
    'name':        ['NAME', 'FULL NAME'],
    'dateOfBirth': ['DOB', 'DATE OF BIRTH', 'BIRTH DATE', 'BORN', 'BIRTH'],
    'address':     ['ADDRESS', 'ADDR', 'RESIDENCE'],
    'idNumber':    ['DL', 'DLN', 'LICENSE NO', 'LICENSE #', 'LICENSE NUMBER',
                    'ID NO', 'ID NUMBER', 'DOCUMENT NO', 'DOCUMENT NUMBER'],
    'expiryDate':  ['EXP', 'EXPIRES', 'EXPIRY', 'EXPIRATION', 'EXP DATE'],
    'issueDate':   ['ISS', 'ISSUED', 'ISSUE DATE', 'ISS DATE'],
    'sex':         ['SEX', 'GENDER'],
}


def load_bytes(req):
    """Return raw image bytes from multipart upload or base64 JSON body."""
    if 'image' in req.files:
        return req.files['image'].read(), None
    elif req.is_json and req.json.get('image'):
        b64 = req.json['image']
        if ',' in b64:
            b64 = b64.split(',', 1)[1]
        return base64.b64decode(b64), None
    return None, "No image provided"


def preprocess(img_array):
    """
    Prepare an ID card photo for OCR:
    1. Upscale to at least 1200px wide — enough resolution for text detection.
    2. CLAHE on the LAB luminance channel — reduces glare and boosts local contrast.
    """
    h, w = img_array.shape[:2]

    if w < 1200:
        scale = 1200 / w
        img_array = cv2.resize(
            img_array,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def convert_mrz_date(yymmdd):
    """Convert MRZ date YYMMDD → MM/DD/YYYY."""
    s = str(yymmdd) if yymmdd else ''
    if not re.match(r'^\d{6}$', s):
        return None
    year = int(s[:2])
    full_year = 1900 + year if year > 30 else 2000 + year
    return f"{s[2:4]}/{s[4:6]}/{full_year}"


def parse_aamva_barcode(raw_bytes):
    """Parse AAMVA PDF417 barcode data into structured fields."""
    try:
        text = raw_bytes.decode('latin-1')
    except Exception:
        return None

    lines = re.split(r'[\r\n]+', text)
    fields = {}
    for line in lines:
        m = re.match(r'^([A-Z]{3})(.+)$', line.strip())
        if m:
            code, value = m.group(1), m.group(2).strip()
            fields[code] = value

    def fmt_date(d):
        """MMDDYYYY → MM/DD/YYYY"""
        if d and re.match(r'^\d{8}$', d):
            return f"{d[0:2]}/{d[2:4]}/{d[4:8]}"
        return d

    sex_map = {'1': 'M', '2': 'F', '9': 'X'}

    # Determine subfile type (DL or ID)
    subfile_type = 'DL' if 'DCA' in fields else 'ID'

    first = fields.get('DCT') or fields.get('DAC')
    last = fields.get('DCS')

    # Some states encode "SMITH,JOHN" in DCT — split it out
    if first and ',' in first and not last:
        parts = first.split(',', 1)
        last = parts[0].strip()
        first = parts[1].strip()

    result = {
        'lastName':     last,
        'firstName':    first,
        'middleName':   fields.get('DAD'),
        'dateOfBirth':  fmt_date(fields.get('DBB')),
        'expiryDate':   fmt_date(fields.get('DBA')),
        'issueDate':    fmt_date(fields.get('DBD')),
        'street':       fields.get('DAG'),
        'city':         fields.get('DAI'),
        'state':        fields.get('DAJ'),
        'zip':          fields.get('DAK'),
        'idNumber':     fields.get('DAQ'),
        'sex':          sex_map.get(fields.get('DBC', ''), fields.get('DBC')),
        'vehicleClass': fields.get('DCA'),
        'subfile_type': subfile_type,
    }
    return result


def decode_pdf417(img_pil):
    """Try to decode PDF417 barcode from image, using three strategies."""
    if not PYZBAR_AVAILABLE:
        return None

    img_array = np.array(img_pil.convert('RGB'))

    def try_decode(arr):
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        codes = pyzbar_decode(gray, symbols=[ZBarSymbol.PDF417])
        if codes:
            return parse_aamva_barcode(codes[0].data)
        return None

    # Strategy 1: full preprocessed image
    result = try_decode(img_array)
    if result:
        print("[Barcode] Decoded from full image")
        return result

    # Strategy 2: bottom-half crop (barcodes usually on lower half of card back)
    h = img_array.shape[0]
    result = try_decode(img_array[h // 2:, :])
    if result:
        print("[Barcode] Decoded from bottom-half crop")
        return result

    # Strategy 3: 2× upscale
    scaled = cv2.resize(img_array, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    result = try_decode(scaled)
    if result:
        print("[Barcode] Decoded from 2x upscale")
        return result

    print("[Barcode] No PDF417 barcode found")
    return None


def collect_words_with_geometry(doctr_result):
    """
    Walk page → block → line → word in docTR result.
    Returns list of dicts with text, conf, and normalized [0,1] bounding box coords.
    """
    words_out = []
    for page in doctr_result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    # docTR geometry: ((x_min, y_min), (x_max, y_max)) normalized
                    geo = word.geometry
                    x_min, y_min = geo[0]
                    x_max, y_max = geo[1]
                    words_out.append({
                        'text': word.value,
                        'conf': word.confidence,
                        'x_min': x_min,
                        'y_min': y_min,
                        'x_max': x_max,
                        'y_max': y_max,
                        'cx': (x_min + x_max) / 2,
                        'cy': (y_min + y_max) / 2,
                    })
    return words_out


def find_label_word(words, aliases):
    """
    Find a label word (or multi-word label) in the word list.
    Returns the last word of the matching span, or None.
    """
    for alias in aliases:
        alias_tokens = alias.upper().split()
        n = len(alias_tokens)
        for i in range(len(words) - n + 1):
            span = [words[i + j]['text'].upper().rstrip(':') for j in range(n)]
            if span == alias_tokens:
                return words[i + n - 1]
    return None


def get_value_words_to_right(words, label_word):
    """Get words to the right of label_word on the same row."""
    label_x_max = label_word['x_max']
    label_cy = label_word['cy']
    candidates = [
        w for w in words
        if w['x_min'] > label_x_max - 0.01
        and abs(w['cy'] - label_cy) < SAME_ROW_Y_TOLERANCE
        and w is not label_word
    ]
    candidates.sort(key=lambda w: w['x_min'])
    return candidates[:6]


def get_value_words_below(words, label_word):
    """Get up to 2 rows of words below the label, x-overlapping with it."""
    label_y_max = label_word['y_max']
    label_x_min = label_word['x_min']
    label_x_max = label_word['x_max']
    candidates = [
        w for w in words
        if w['y_min'] > label_y_max - 0.01
        and w['x_min'] < label_x_max + SAME_COL_X_TOLERANCE
        and w['x_max'] > label_x_min - SAME_COL_X_TOLERANCE
    ]
    if not candidates:
        return []

    # Group into rows by proximity
    candidates.sort(key=lambda w: w['cy'])
    rows = []
    current_row = []
    current_y = None
    for w in candidates:
        if current_y is None or abs(w['cy'] - current_y) < SAME_ROW_Y_TOLERANCE:
            current_row.append(w)
            current_y = w['cy'] if current_y is None else (current_y + w['cy']) / 2
        else:
            rows.append(current_row)
            current_row = [w]
            current_y = w['cy']
            if len(rows) >= 2:
                break
    if current_row and len(rows) < 2:
        rows.append(current_row)

    result = []
    for row in rows[:2]:
        result.extend(row[:8])
    return result


def extract_fields_spatially(words):
    """Use label-proximity to extract structured fields from word geometry."""
    extracted = {}
    for field, aliases in LABEL_ALIASES.items():
        label_word = find_label_word(words, aliases)
        if not label_word:
            continue
        # Try right first, then below
        value_words = get_value_words_to_right(words, label_word)
        if not value_words:
            value_words = get_value_words_below(words, label_word)
        if value_words:
            value = ' '.join(w['text'] for w in value_words).strip()
            if value:
                extracted[field] = value

    # If no full name but first/last found, merge them
    if 'name' not in extracted and ('firstName' in extracted or 'lastName' in extracted):
        parts = [extracted.get('firstName', ''), extracted.get('lastName', '')]
        merged = ' '.join(p for p in parts if p).strip()
        if merged:
            extracted['name'] = merged

    print(f"[Spatial] Extracted: {extracted}")
    return extracted


def detect_document_type(mrz_data, barcode_data):
    """Detect document type from MRZ or barcode data."""
    if mrz_data:
        doc_type = str(mrz_data.get('type', '') or '').upper()
        if doc_type.startswith('P'):
            return 'passport'
        if doc_type.startswith(('I', 'A', 'C')):
            return 'id_card'
        if doc_type.startswith('V'):
            return 'travel_document'
        # Fallback: check MRZ line lengths from raw_text
        mrz_raw = mrz_data.get('raw_text', '') or ''
        lines = [l for l in mrz_raw.split('\n') if l.strip()]
        if lines and len(lines[0]) == 44:
            return 'passport'
        if lines and len(lines[0]) == 30:
            return 'id_card'
        return 'id_card'  # default for any MRZ document

    if barcode_data:
        if barcode_data.get('vehicleClass') or barcode_data.get('subfile_type') == 'DL':
            return 'dl'
        return 'state_id'

    return 'unknown'


@app.route('/health', methods=['GET'])
def health():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    return jsonify({
        "status": "ok",
        "ocr": "doctr",
        "device": device,
        "passporteye": PASSPORTEYE_AVAILABLE,
        "pyzbar": PYZBAR_AVAILABLE,
    })


@app.route('/ocr', methods=['POST'])
def ocr():
    file_bytes, err = load_bytes(request)
    if err:
        return jsonify({"error": err}), 400

    side = None
    if request.is_json:
        side = request.json.get('side')  # 'front' | 'back' | None

    print(f"\n[OCR] Received image: {len(file_bytes) / 1024:.1f} KB | side={side}")

    img_pil = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    img_array = np.array(img_pil)
    print(f"[docTR] Original size: {img_pil.size}")

    # ── docTR (front or unspecified) ──────────────────────────────────────────
    spatial_fields = {}
    lines_out = []
    avg_confidence = 0.0
    raw_text = ''

    if side != 'back':
        img_preprocessed = preprocess(img_array)
        print(f"[docTR] Preprocessed size: {img_preprocessed.shape[1]}x{img_preprocessed.shape[0]}")

        doc = DocumentFile.from_images([img_preprocessed])
        result = model(doc)

        words_geo = collect_words_with_geometry(result)
        spatial_fields = extract_fields_spatially(words_geo)

        total_confidence = 0.0
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    words = line.words
                    if not words:
                        continue
                    line_text = " ".join(w.value for w in words).strip()
                    line_conf = sum(w.confidence for w in words) / len(words)
                    if line_text:
                        lines_out.append({"text": line_text, "confidence": round(line_conf, 4)})
                        total_confidence += line_conf

        avg_confidence = (total_confidence / len(lines_out) * 100) if lines_out else 0.0
        raw_text = "\n".join(item["text"] for item in lines_out)
        print(f"[docTR] Found {len(lines_out)} lines | avg confidence: {avg_confidence:.1f}%")
        print(f"[docTR] Raw text:\n{raw_text}\n")

    # ── PDF417 barcode (back or unspecified) ──────────────────────────────────
    barcode_data = None
    if side != 'front' and PYZBAR_AVAILABLE:
        print("[Barcode] Scanning for PDF417...")
        barcode_data = decode_pdf417(img_pil)
        if barcode_data:
            print(f"[Barcode] Found: {barcode_data}")

    # ── PassportEye MRZ (front or unspecified) ────────────────────────────────
    mrz_data = None
    if side != 'back' and PASSPORTEYE_AVAILABLE:
        print("[PassportEye] Scanning for MRZ zone...")
        try:
            mrz = read_mrz(io.BytesIO(file_bytes))
            if mrz:
                mrz_data = mrz.to_dict()
                print(f"[PassportEye] MRZ found: {mrz_data}")
            else:
                print("[PassportEye] No MRZ zone detected")
        except Exception as e:
            print(f"[PassportEye] Error: {e}")

    # ── Document type detection ────────────────────────────────────────────────
    doc_type = detect_document_type(mrz_data, barcode_data)
    print(f"[DocType] Detected: {doc_type}")

    # ── Merge fields: barcode > MRZ > spatial ─────────────────────────────────
    fields = {
        'name': None, 'dateOfBirth': None, 'address': None,
        'idNumber': None, 'expiryDate': None, 'issueDate': None,
        'sex': None, 'state': None,
    }

    # Spatial (lowest priority)
    if spatial_fields:
        fields.update({
            'name':        spatial_fields.get('name'),
            'dateOfBirth': spatial_fields.get('dateOfBirth'),
            'address':     spatial_fields.get('address'),
            'idNumber':    spatial_fields.get('idNumber'),
            'expiryDate':  spatial_fields.get('expiryDate'),
            'issueDate':   spatial_fields.get('issueDate'),
            'sex':         spatial_fields.get('sex'),
        })

    # MRZ overrides spatial
    if mrz_data:
        surname = (mrz_data.get('surname') or '').replace('<', ' ').strip()
        given_names = (mrz_data.get('given_names') or '').replace('<', ' ').strip()
        fields.update({
            'name':        ' '.join(filter(None, [given_names, surname])) or None,
            'dateOfBirth': convert_mrz_date(mrz_data.get('date_of_birth')),
            'idNumber':    (mrz_data.get('document_number') or '').replace('<', '').strip() or None,
            'state':       mrz_data.get('country') or mrz_data.get('nationality'),
        })

    # Barcode overrides everything
    if barcode_data:
        addr_parts = [
            barcode_data.get('street') or '',
            barcode_data.get('city') or '',
            barcode_data.get('state') or '',
            barcode_data.get('zip') or '',
        ]
        address = ', '.join(p for p in addr_parts if p) or None
        first = barcode_data.get('firstName') or ''
        last = barcode_data.get('lastName') or ''
        middle = barcode_data.get('middleName') or ''
        name = ' '.join(filter(None, [first, middle, last])) or None
        fields.update({
            'name':        name,
            'dateOfBirth': barcode_data.get('dateOfBirth'),
            'address':     address,
            'idNumber':    barcode_data.get('idNumber'),
            'expiryDate':  barcode_data.get('expiryDate'),
            'issueDate':   barcode_data.get('issueDate'),
            'sex':         barcode_data.get('sex'),
            'state':       barcode_data.get('state'),
        })

    print(f"[OCR] Done — fields: {fields}\n")

    return jsonify({
        "success": True,
        "documentType": doc_type,
        "side": side,
        "fields": fields,
        "confidence": round(avg_confidence, 2),
        "raw_text": raw_text,
        "lines": lines_out,
        "mrz": mrz_data,
        "barcode": barcode_data,
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3002, debug=False)
