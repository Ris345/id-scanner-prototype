import io
import re
import os
import base64
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

# PassportEye reads the MRZ (machine-readable zone) on passports — those two
# lines of weird characters at the bottom. Needs the Tesseract CLI installed.
try:
    from passporteye import read_mrz
    PASSPORTEYE_AVAILABLE = True
except ImportError:
    PASSPORTEYE_AVAILABLE = False
    print("PassportEye not available — install passporteye + tesseract")

# pyzbar decodes barcodes. We use it specifically for PDF417 — the 2D barcode
# on the back of every US driver's license. It contains ALL the card fields
# perfectly structured (no OCR guessing required). Needs libzbar on the OS.
try:
    from pyzbar.pyzbar import decode as pyzbar_decode, ZBarSymbol
    PYZBAR_AVAILABLE = True
except Exception as e:
    PYZBAR_AVAILABLE = False
    print(f"pyzbar not available ({type(e).__name__}: {e}) — run: brew install zbar")

# scikit-image Sauvola adaptive binarization (PaddleOCR preprocessing).
try:
    from skimage.filters import threshold_sauvola
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("scikit-image not available — pip install scikit-image")

import torch
from doctr.models import ocr_predictor
from doctr.io import DocumentFile

# ── PaddleOCR PP-OCRv5 — supplemental recognition pass ───────────────────────
# Slots between docTR spatial extraction and regex fallback.
# Triggered only when docTR confidence < 0.75 for name, dateOfBirth, or idNumber.
# Install (CPU):  pip install paddlepaddle paddleocr
# Install (GPU):  pip install paddlepaddle-gpu paddleocr
PADDLE_ENABLED = os.environ.get('PADDLE_ENABLED', 'true').lower() != 'false'
PADDLE_AVAILABLE = False
paddle_ocr = None

if PADDLE_ENABLED:
    try:
        from paddleocr import PaddleOCR as _PaddleOCR
        PADDLE_AVAILABLE = True
    except Exception as _e:
        print(f"[Paddle] Import failed ({_e}) — falling through to regex silently")

app = Flask(__name__)

# ── GPU setup ─────────────────────────────────────────────────────────────────
# Pick the best available hardware. On your M4 Mac this will be 'mps' (Metal).
# CUDA = NVIDIA GPU, MPS = Apple Silicon GPU, CPU = no GPU found.
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("[Device] Using CUDA GPU")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print("[Device] Using Apple MPS (Metal) — GPU acceleration enabled")
else:
    device = torch.device('cpu')
    print("[Device] Using CPU")

# Load the OCR model and move it onto whichever device we picked above.
# db_resnet50 = detects where the text boxes are.
# parseq      = reads the actual characters inside those boxes (transformer-based).
print("Loading docTR model (db_resnet50 + parseq)...")
model = ocr_predictor(det_arch='db_resnet50', reco_arch='parseq', pretrained=True)
model = model.to(device)
print(f"docTR ready on {device}.")

# ── PaddleOCR model init ───────────────────────────────────────────────────────
# PP-OCRv5 server model on CUDA (higher accuracy), mobile model on CPU/MPS.
# PaddleOCR has no Apple MPS backend — runs on CPU on Apple Silicon automatically.
if PADDLE_AVAILABLE:
    # PaddleOCR 3.x removed use_gpu (device is auto-detected) and renamed
    # use_angle_cls → use_textline_orientation.
    _paddle_model = 'PP-OCRv5_server' if device.type == 'cuda' else 'PP-OCRv5_mobile'
    print(f"[Paddle] Loading PaddleOCR ({_paddle_model})...")
    try:
        paddle_ocr = _PaddleOCR(
            use_textline_orientation=True,  # handles rotated phone photos
        )
        print("[Paddle] PaddleOCR ready.")
    except Exception as _e:
        print(f"[Paddle] Failed to load model: {_e} — will fall through to regex silently")
        PADDLE_AVAILABLE = False

# ── Ollama gap-filler config ──────────────────────────────────────────────────
# Ollama runs on the host machine. Inside Docker, host.docker.internal resolves
# to the Mac's localhost. For local non-Docker runs, falls back to localhost.
OLLAMA_ENABLED = os.environ.get('OLLAMA_ENABLED', 'true').lower() != 'false'
OLLAMA_HOST    = os.environ.get('OLLAMA_HOST', 'host.docker.internal')
OLLAMA_MODEL   = os.environ.get('OLLAMA_MODEL', 'gemma3:4b')
OLLAMA_URL     = f'http://{OLLAMA_HOST}:11434/api/generate'
print(f'[Ollama] {"enabled" if OLLAMA_ENABLED else "disabled"}  model={OLLAMA_MODEL}  url={OLLAMA_URL}')

# ── Spatial extraction constants ───────────────────────────────────────────────
# docTR returns bounding box coordinates as fractions of the page (0.0 → 1.0).
# These tolerances define how close a word needs to be to count as "same row" or
# "same column" as a label word. 0.03 = 3% of page height, which is about one
# line of text on a standard ID card.
SAME_ROW_Y_TOLERANCE = 0.03
SAME_COL_X_TOLERANCE = 0.05

# Maps the field names we want to extract → all the label words that could appear
# on a real ID card for that field. E.g. DOB/DATE OF BIRTH/BORN all mean dateOfBirth.
LABEL_ALIASES = {
    'lastName':    ['LN', 'LAST', 'LAST NAME', 'SURNAME'],
    'firstName':   ['FN', 'FIRST', 'FIRST NAME', 'GIVEN', 'GIVEN NAME', 'GIVEN NAMES', 'FORENAME', 'FNAME', 'F NAME'],
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
    """
    Pull the raw image bytes out of the incoming HTTP request.
    Supports two ways the caller can send an image:
      1. Multipart form upload (req.files['image']) — standard file upload
      2. JSON body with base64-encoded string (req.json['image']) — what the app sends
    Returns (bytes, None) on success or (None, error_message) on failure.
    """
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
    Prepare an ID card photo for docTR OCR. Two things happen here:

    1. Upscale to at least 1200px wide.
       docTR needs enough pixels to detect small text. Most phone photos are
       fine, but low-res uploads would fail without this.

    2. CLAHE (Contrast Limited Adaptive Histogram Equalization).
       This boosts local contrast to handle glare, shadows, and washed-out areas
       common in ID photos. We apply it only to the luminance (L) channel in LAB
       color space so colors don't shift.
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


# ── PaddleOCR preprocessing pipeline ─────────────────────────────────────────

def _order_points(pts):
    """Order 4 corner points as [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left: smallest sum
    rect[2] = pts[np.argmax(s)]   # bottom-right: largest sum
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right: smallest diff
    rect[3] = pts[np.argmax(diff)]  # bottom-left: largest diff
    return rect


def correct_perspective(img_array):
    """
    Detect the ID card's corners and warp the image to a flat top-down view.
    Finds the largest quadrilateral contour in the edge map and applies
    getPerspectiveTransform + warpPerspective. Returns the original if no
    four-corner shape is found (non-destructive fallback).
    """
    gray    = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 50, 150)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges   = cv2.dilate(edges, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_array

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    card_quad = None
    for c in contours[:5]:
        peri  = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            card_quad = approx
            break

    if card_quad is None:
        return img_array

    pts  = card_quad.reshape(4, 2).astype(np.float32)
    rect = _order_points(pts)
    tl, tr, br, bl = rect

    width  = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 100 or height < 50:
        return img_array

    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                   dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img_array, M, (width, height))


def mask_specular_glare(img_array):
    """
    Detect blown-out specular highlights and inpaint them.
    Pixels with luminance > 240 are masked; INPAINT_TELEA fills them in
    (radius=7). Returns (corrected_array, glare_ratio) where glare_ratio is
    the fraction (0.0–1.0) of blown-out pixels in the original image.
    """
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    glare_ratio = float(np.count_nonzero(mask)) / mask.size

    if glare_ratio < 0.001:
        return img_array, glare_ratio

    # Dilate slightly so inpainting covers the halo around each bright spot
    kernel       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)

    bgr       = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(bgr, mask_dilated, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    return cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB), glare_ratio


def adaptive_binarize(img_array):
    """
    Binarize via Sauvola's method — threshold adapts per local window, handling
    uneven illumination and shadows far better than global Otsu.
    Returns a 3-channel uint8 image (PaddleOCR expects BGR/RGB arrays, not masks).
    Falls back to plain grayscale→RGB if scikit-image is not installed.
    """
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    if SKIMAGE_AVAILABLE:
        thresh  = threshold_sauvola(gray, window_size=25)
        binary  = (gray > thresh).astype(np.uint8) * 255
    else:
        binary = gray   # plain gray is better than nothing

    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)


def compute_glare_ratio(img_array):
    """Return fraction of pixels with luminance > 240 (blown-out highlights)."""
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    return float(np.count_nonzero(gray > 240)) / gray.size


def preprocess_for_paddle(img_array):
    """
    Preprocessing pipeline for PaddleOCR PP-OCRv5:
      1. Perspective correction — straighten tilted/angled card
      2. Specular glare masking — inpaint blown-out highlights

    Intentionally NO binarization: PP-OCRv5 is a deep learning model trained
    on natural images. Binarization helps classical engines (Tesseract) but
    destroys the color/gradient cues that neural models rely on, hurting accuracy.
    Returns (preprocessed_array, glare_ratio).
    """
    img              = correct_perspective(img_array)
    img, glare_ratio = mask_specular_glare(img)
    return img, glare_ratio


def run_paddle_ocr(img_array):
    """
    Run PaddleOCR PP-OCRv5 on a preprocessed image and extract ID fields.

    Called only when docTR confidence < 0.75 for name, dateOfBirth, or idNumber.
    Reuses extract_fields_regex() on the PaddleOCR text so field logic stays
    in one place.

    Returns {fieldName: {"value": str, "confidence": float, "source": "paddle"}}
    for every field found, or {} on any failure.
    """
    if not PADDLE_AVAILABLE or paddle_ocr is None:
        return {}

    try:
        result = paddle_ocr.ocr(img_array)
    except Exception as e:
        print(f"[Paddle] Inference error: {e}")
        return {}

    if not result or not result[0]:
        return {}

    # Flatten to list of (text, confidence) tuples
    lines_text = []
    for item in result[0]:
        text = item[1][0]
        conf = float(item[1][1])
        lines_text.append((text, conf))

    if not lines_text:
        return {}

    raw_text   = "\n".join(t for t, _ in lines_text)
    text_lines = [t for t, _ in lines_text]
    avg_conf   = sum(c for _, c in lines_text) / len(lines_text)

    print(f"[Paddle] Recognized {len(lines_text)} lines, avg_conf={avg_conf:.3f}")
    print(f"[Paddle] Raw text:\n{raw_text}")

    # Reuse the regex extractor — it works on any flat OCR text
    flat_fields = _extract_fields_regex_flat(raw_text, text_lines)

    extracted = {}
    for key, val in flat_fields.items():
        if val:
            extracted[key] = {"value": val, "confidence": round(avg_conf, 4), "source": "paddle"}

    print(f"[Paddle] Extracted fields: {list(extracted.keys())}")
    return extracted


# ── End PaddleOCR section ─────────────────────────────────────────────────────

def convert_mrz_date(yymmdd):
    """
    Convert a MRZ date string (YYMMDD) into MM/DD/YYYY format.
    MRZ dates use only 2-digit years, so we guess the century:
      - year > 30 → 1900s (e.g. "85" → 1985)
      - year <= 30 → 2000s (e.g. "28" → 2028)
    Returns None if the input doesn't look like 6 digits.
    """
    s = str(yymmdd) if yymmdd else ''
    if not re.match(r'^\d{6}$', s):
        return None
    year = int(s[:2])
    full_year = 1900 + year if year > 30 else 2000 + year
    return f"{s[2:4]}/{s[4:6]}/{full_year}"


def parse_aamva_barcode(raw_bytes):
    """
    Parse the AAMVA (American Association of Motor Vehicle Administrators)
    standard data from a decoded PDF417 barcode.

    The barcode is a plain-text blob where every record starts with a 3-letter
    code. For example:
      DCS = last name
      DCT = first name
      DBB = date of birth
      DBA = expiry date
      DAQ = ID / license number
      DAJ = state
      DBC = sex (1=M, 2=F, 9=X)
      DCA = vehicle class (only present on driver's licenses, not state IDs)

    Returns a dict with all the fields, or None if parsing blows up.
    """
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
        if d and re.match(r'^\d{8}$', d):
            return f"{d[0:2]}/{d[2:4]}/{d[4:8]}"
        return d

    sex_map = {'1': 'M', '2': 'F', '9': 'X'}
    subfile_type = 'DL' if 'DCA' in fields else 'ID'

    first = fields.get('DCT') or fields.get('DAC')
    last  = fields.get('DCS')

    if first and ',' in first and not last:
        parts = first.split(',', 1)
        last  = parts[0].strip()
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
    """
    Try to decode the PDF417 barcode from an ID card back image.
    Three strategies in order: full image → bottom-half crop → 2× upscale.
    Returns the parsed AAMVA dict, or None if nothing decoded.
    """
    if not PYZBAR_AVAILABLE:
        return None

    img_array = np.array(img_pil.convert('RGB'))

    def try_decode(arr):
        gray  = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        codes = pyzbar_decode(gray, symbols=[ZBarSymbol.PDF417])
        if codes:
            return parse_aamva_barcode(codes[0].data)
        return None

    result = try_decode(img_array)
    if result:
        print("[Barcode] Decoded from full image")
        return result

    h = img_array.shape[0]
    result = try_decode(img_array[h // 2:, :])
    if result:
        print("[Barcode] Decoded from bottom-half crop")
        return result

    scaled = cv2.resize(img_array, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    result = try_decode(scaled)
    if result:
        print("[Barcode] Decoded from 2x upscale")
        return result

    print("[Barcode] No PDF417 barcode found")
    return None


def collect_words_with_geometry(doctr_result):
    """
    Walk the docTR result tree and collect every recognized word along with its
    bounding box on the page.

    Returns a list of dicts: text, conf, x_min, y_min, x_max, y_max, cx, cy.
    All coordinates are normalized (0.0–1.0).
    """
    words_out = []
    for page in doctr_result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    geo   = word.geometry
                    x_min, y_min = geo[0]
                    x_max, y_max = geo[1]
                    words_out.append({
                        'text':  word.value,
                        'conf':  word.confidence,
                        'x_min': x_min,
                        'y_min': y_min,
                        'x_max': x_max,
                        'y_max': y_max,
                        'cx':    (x_min + x_max) / 2,
                        'cy':    (y_min + y_max) / 2,
                    })
    return words_out


def _edit_distance(a, b):
    """Levenshtein distance between two strings (fast, O(n) space)."""
    if abs(len(a) - len(b)) > 2:
        return 99
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp  = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev  = temp
    return dp[n]


def _strip_label_punct(token):
    """Strip surrounding punctuation from an OCR'd label token before matching."""
    return re.sub(r"^[^A-Z0-9]+|[^A-Z0-9]+$", '', token.upper())


def find_label_word(words, aliases):
    """
    Search the word list for a label word (or multi-word label phrase).
    Pass 1: exact match with punctuation stripped.
    Pass 2: fuzzy match (edit distance ≤ 1) for short single-token aliases only.
    Returns the matching word dict, or None.
    """
    cleaned = [_strip_label_punct(w['text']) for w in words]

    for alias in aliases:
        alias_tokens = alias.upper().split()
        n = len(alias_tokens)
        for i in range(len(words) - n + 1):
            span = [cleaned[i + j] for j in range(n)]
            if span == alias_tokens:
                return words[i + n - 1]

    for alias in aliases:
        alias_tokens = alias.upper().split()
        if len(alias_tokens) != 1:
            continue
        target = alias_tokens[0]
        if len(target) > 6:
            continue
        for i, w in enumerate(words):
            if _edit_distance(cleaned[i], target) <= 1:
                return w

    return None


def get_value_words_to_right(words, label_word):
    """
    Find words to the RIGHT of the label on the same row.
    Returns up to 6 words sorted left-to-right.
    """
    label_x_max = label_word['x_max']
    label_cy    = label_word['cy']
    candidates  = [
        w for w in words
        if w['x_min'] > label_x_max - 0.01
        and abs(w['cy'] - label_cy) < SAME_ROW_Y_TOLERANCE
        and w is not label_word
    ]
    candidates.sort(key=lambda w: w['x_min'])
    return candidates[:6]


def get_value_words_below(words, label_word):
    """
    Find words BELOW the label and roughly in the same horizontal column.
    Returns up to 2 rows, 8 words each.
    """
    label_y_max = label_word['y_max']
    label_x_min = label_word['x_min']
    label_x_max = label_word['x_max']
    candidates  = [
        w for w in words
        if w['y_min'] > label_y_max - 0.01
        and w['x_min'] < label_x_max + SAME_COL_X_TOLERANCE
        and w['x_max'] > label_x_min - SAME_COL_X_TOLERANCE
    ]
    if not candidates:
        return []

    candidates.sort(key=lambda w: w['cy'])
    rows        = []
    current_row = []
    current_y   = None
    for w in candidates:
        if current_y is None or abs(w['cy'] - current_y) < SAME_ROW_Y_TOLERANCE:
            current_row.append(w)
            current_y = w['cy'] if current_y is None else (current_y + w['cy']) / 2
        else:
            rows.append(current_row)
            current_row = [w]
            current_y   = w['cy']
            if len(rows) >= 2:
                break
    if current_row and len(rows) < 2:
        rows.append(current_row)

    result = []
    for row in rows[:2]:
        result.extend(row[:8])
    return result


def _clean_name_value(val):
    """Strip OCR noise (leading/trailing punctuation) from a name value."""
    val = re.sub(r"^[^A-Za-z\-']+|[^A-Za-z\-']+$", '', val)
    return re.sub(r'\s+', ' ', val).strip()


def extract_fields_spatially(words):
    """
    Main spatial extraction pass — label-proximity bounding-box search.

    For each field, finds the label word then grabs value words to its right
    or below it. Returns:
      {fieldName: {"value": str, "confidence": float, "source": "doctr"}}

    Confidence is the average word-level confidence of the extracted value words.
    """
    NAME_FIELDS = ('firstName', 'lastName', 'name')
    extracted   = {}

    for field, aliases in LABEL_ALIASES.items():
        label_word = find_label_word(words, aliases)

        # ── Name-field diagnostics ────────────────────────────────────────────
        # Printed for every name-related field so you can see exactly what
        # docTR read, whether the label was found, and what candidates exist.
        if field in NAME_FIELDS:
            if label_word:
                print(f'[PY][Name] field={field!r}  label found: '
                      f'"{label_word["text"]}"  '
                      f'pos=({label_word["cx"]:.3f},{label_word["cy"]:.3f})  '
                      f'conf={label_word["conf"]:.3f}')
                right = get_value_words_to_right(words, label_word)
                below = get_value_words_below(words, label_word)
                print(f'[PY][Name]   right candidates : {[w["text"] for w in right]}')
                print(f'[PY][Name]   below candidates : {[w["text"] for w in below[:6]]}')
            else:
                print(f'[PY][Name] field={field!r}  NO LABEL FOUND  '
                      f'(searched aliases: {aliases})')
                # Show all OCR tokens to help diagnose what was actually read
                all_tokens = [w['text'] for w in words]
                print(f'[PY][Name]   all docTR tokens: {all_tokens}')

        if not label_word:
            continue

        value_words = get_value_words_to_right(words, label_word)
        if not value_words:
            value_words = get_value_words_below(words, label_word)

        if value_words:
            value = ' '.join(w['text'] for w in value_words).strip()
            value = _clean_name_value(value) if field in NAME_FIELDS else value
            if value:
                conf = sum(w['conf'] for w in value_words) / len(value_words)
                extracted[field] = {"value": value, "confidence": round(conf, 4), "source": "doctr"}
                if field in NAME_FIELDS:
                    print(f'[PY][Name]   => extracted {field!r}: "{value}"  conf={conf:.3f}')

    # Merge first + last into a single 'name' field if we didn't find 'name' directly
    if 'name' not in extracted and ('firstName' in extracted or 'lastName' in extracted):
        fn_entry = extracted.get('firstName') or {}
        ln_entry = extracted.get('lastName') or {}
        parts    = [fn_entry.get('value', ''), ln_entry.get('value', '')]
        merged   = ' '.join(p for p in parts if p).strip()
        if merged:
            conf = (fn_entry.get('confidence', 0.0) + ln_entry.get('confidence', 0.0)) / 2
            extracted['name'] = {"value": merged, "confidence": round(conf, 4), "source": "doctr"}
            print(f'[PY][Name]   => merged name: "{merged}"  conf={conf:.3f}')

    if 'name' not in extracted:
        print(f'[PY][Name] *** name field still empty after spatial pass ***')

    print(f"[PY][Spatial] Extracted: { {k: v['value'] for k, v in extracted.items()} }")
    return extracted


def _extract_fields_regex_flat(raw_text, lines):
    """
    Core regex extraction logic — returns a plain {fieldName: value_string} dict.
    Called by both extract_fields_regex() and run_paddle_ocr().
    """
    result = {}

    def val_same_or_next(i, strip_pat):
        after = re.sub(strip_pat, '', lines[i], flags=re.I).strip()
        if after:
            return after
        if i + 1 < len(lines):
            return lines[i + 1].strip()
        return ''

    # ── Date of birth ──────────────────────────────────────────────────────────
    date_pat  = re.compile(r'\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})\b')
    dob_label = re.compile(r'\b(DOB|DATE\s+OF\s+BIRTH|BIRTH\s+DATE|BORN)\b', re.I)
    for i, line in enumerate(lines):
        if dob_label.search(line):
            candidate = val_same_or_next(i, r'^.*?(DOB|DATE\s+OF\s+BIRTH|BIRTH\s+DATE|BORN)[:\s]*')
            m = date_pat.search(candidate)
            if m:
                result['dateOfBirth'] = m.group(0)
                break
    if 'dateOfBirth' not in result:
        m = date_pat.search(raw_text)
        if m:
            result['dateOfBirth'] = m.group(0)

    # ── Expiry date ────────────────────────────────────────────────────────────
    exp_label = re.compile(r'\b(EXP|EXPIRES|EXPIRY|EXPIRATION)\b', re.I)
    for i, line in enumerate(lines):
        if exp_label.search(line):
            candidate = val_same_or_next(i, r'^.*?(EXP|EXPIRES|EXPIRY|EXPIRATION)[:\s]*')
            m = date_pat.search(candidate)
            if m:
                result['expiryDate'] = m.group(0)
                break

    # ── Issue date ─────────────────────────────────────────────────────────────
    iss_label = re.compile(r'\b(ISS|ISSUED|ISSUE\s+DATE)\b', re.I)
    for i, line in enumerate(lines):
        if iss_label.search(line):
            candidate = val_same_or_next(i, r'^.*?(ISS|ISSUED|ISSUE\s+DATE)[:\s]*')
            m = date_pat.search(candidate)
            if m:
                result['issueDate'] = m.group(0)
                break

    # ── ID / licence number ────────────────────────────────────────────────────
    id_label = re.compile(r'\b(DL|DLN|LICENSE|LIC|ID\s*NO|DOCUMENT\s*NO|ID#)\b', re.I)
    id_val   = re.compile(r'[A-Z]?\d{5,}|[A-Z][A-Z0-9]{6,14}', re.I)
    for i, line in enumerate(lines):
        if id_label.search(line):
            candidate = val_same_or_next(i, r'^.*?(DL|DLN|LICENSE|LIC|ID\s*NO|DOCUMENT\s*NO|ID#)[:\s]*')
            m = id_val.search(candidate)
            if m:
                result['idNumber'] = m.group(0)
                break
    if 'idNumber' not in result:
        for line in lines:
            m = re.search(r'\b[A-Z]\d{7,}\b', line)
            if m:
                result['idNumber'] = m.group(0)
                break

    # ── Name ───────────────────────────────────────────────────────────────────
    name_like   = re.compile(r"^[A-Za-z\s\-']+$")
    header_skip = re.compile(
        r'^(DRIVER\s*(LICENSE|LICENCE)|IDENTIFICATION\s*CARD|ID\s*CARD|CLASS\s+[A-Z]|'
        r'NOT\s+FOR\s+FEDERAL|FEDERAL\s+PURPOSES|DOB|SEX|ADDRESS|LICENSE|EXP|ISS|'
        r'HT|WT|EYES|EYE|HAIR|STATE|DL|DLN|ORGAN|DONOR|VETERAN|REAL\s*ID|'
        r'ENDORSEMENTS|RESTRICTIONS|UNDER|NONE)$', re.I)

    first_name = last_name = None
    for i, line in enumerate(lines):
        if re.search(r'\b(NAME|FULL\s+NAME)\b', line, re.I):
            val = val_same_or_next(i, r'^.*?(NAME|FULL\s+NAME)[:\s]*')
            if val and name_like.match(val) and 2 <= len(val) <= 50:
                result['name'] = val
                break
        if re.search(r'\b(FIRST\s+NAME|GIVEN\s+NAMES?|FORENAME|FNAME|F\s+NAME|FIRST|GIVEN|FN)\b', line, re.I):
            val = val_same_or_next(i, r'^.*?(FIRST\s+NAME|GIVEN\s+NAMES?|FORENAME|FNAME|F\s+NAME|FIRST|GIVEN|FN)[.:\s]*')
            val = _clean_name_value(val)
            if val and name_like.match(val) and len(val) >= 2:
                first_name = val
        if re.search(r'\b(LAST|SURNAME|LN)\b', line, re.I):
            val = val_same_or_next(i, r'^.*?(LAST|SURNAME|LN)[:\s]*')
            if val and name_like.match(val) and len(val) >= 2:
                last_name = val

    if 'name' not in result:
        if first_name or last_name:
            result['name'] = ' '.join(filter(None, [first_name, last_name]))
        else:
            for line in lines:
                if re.search(r'\d', line) or len(line) < 4:
                    continue
                if header_skip.match(line.strip()):
                    continue
                words = [w for w in line.split() if re.match(r'^[A-Za-z]+$', w) and len(w) > 1]
                if 2 <= len(words) <= 4:
                    result['name'] = ' '.join(words)
                    break

    # ── Address ────────────────────────────────────────────────────────────────
    street_pat = re.compile(
        r'\d+\s+[\w\s]+(ST|STREET|AVE|AVENUE|RD|ROAD|DR|DRIVE|LN|LANE|BLVD|CT|COURT|WAY|PL|PLACE)\b', re.I)
    zip_pat    = re.compile(r'\b\d{5}(-\d{4})?\b')
    addr_lines = []
    capturing  = False
    for line in lines:
        if re.search(r'\b(ADDRESS|ADDR)\b', line, re.I):
            capturing = True
            after = re.sub(r'.*?(ADDRESS|ADDR)[:\s]*', '', line, flags=re.I).strip()
            if after:
                addr_lines.append(after)
            continue
        if capturing or street_pat.search(line):
            if len(line) > 5:
                addr_lines.append(line)
                capturing = True
            if zip_pat.search(line):
                break
    if addr_lines:
        result['address'] = ', '.join(addr_lines)

    # ── Sex ────────────────────────────────────────────────────────────────────
    for i, line in enumerate(lines):
        if re.search(r'\bSEX\b', line, re.I):
            candidate = val_same_or_next(i, r'^.*?SEX[:\s]*')
            m = re.search(r'\b(MALE|FEMALE|[MFX])\b', candidate, re.I)
            if m:
                v = m.group(0).upper()
                result['sex'] = 'M' if v == 'MALE' else 'F' if v == 'FEMALE' else v
                break

    # ── Label-free pattern fallback ────────────────────────────────────────────

    # ID number: letter + 6–9 digits (most states)
    if 'idNumber' not in result:
        for line in lines:
            m = re.search(r'\b([A-Z]\d{6,9})\b', line)
            if m:
                result['idNumber'] = m.group(1)
                break

    # ID number: pure numeric 7–12 digits (NY and other no-letter states).
    # Also recovers docTR-split numbers — e.g. "0210 049 849" → "0210049849".
    if 'idNumber' not in result:
        for line in lines:
            digit_tokens = re.findall(r'\b(\d+)\b', line)
            joined = ''.join(digit_tokens)
            if 7 <= len(joined) <= 12 and len(digit_tokens) >= 2:
                result['idNumber'] = joined
                print(f'[PY][Regex] idNumber joined from {digit_tokens} → {joined}')
                break

    if 'dateOfBirth' not in result or 'expiryDate' not in result or 'issueDate' not in result:
        import datetime
        current_year = datetime.datetime.now().year
        all_dates    = re.findall(r'\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})\b', raw_text)
        parsed_dates = []
        for mo, dy, yr in all_dates:
            year = int(yr) if len(yr) == 4 else (2000 + int(yr) if int(yr) < 30 else 1900 + int(yr))
            parsed_dates.append((f"{mo}/{dy}/{yr}", year))

        for date_str, year in parsed_dates:
            if 'dateOfBirth' not in result and year <= current_year - 16:
                result['dateOfBirth'] = date_str
            elif 'expiryDate' not in result and year >= current_year:
                result['expiryDate'] = date_str
            elif 'issueDate' not in result and current_year - 10 <= year < current_year:
                result['issueDate'] = date_str

    # Name — Pass 1: labeled or multi-word alpha line (standard layout)
    if 'name' not in result:
        skip = re.compile(
            r'^(CALIFORNIA|FLORIDA|TEXAS|NEW\s+YORK|NEW|YORK|DRIVER|LICENSE|LICENCE|'
            r'IDENTIFICATION|ID\s*CARD|CLASS|NOT\s+FOR|FEDERAL|REAL\s*ID|DOB|SEX|EXP|'
            r'ISS|HT|WT|EYES?|HAIR|STATE|DL|DLN|ORGAN|DONOR|VETERAN|NONE|UNDER|PERMIT|'
            r'LEARNER|PURPOSES|ENDORSEMENTS|RESTRICTIONS|[A-Z]{2}\s+\d|DD\s).*$', re.I)
        for line in lines:
            if re.search(r'\d', line) or len(line) < 4:
                continue
            if skip.match(line.strip()):
                continue
            words = [w for w in line.split() if re.match(r'^[A-Za-z\-\']+$', w) and len(w) > 1]
            if 2 <= len(words) <= 5:
                result['name'] = ' '.join(words)
                break

    # Name — Pass 2: consecutive ALL-CAPS single-word lines (NY / no-label layout).
    # Line N = LASTNAME, line N+1 = FIRSTNAME[,MIDDLENAME]
    # e.g. "ACHARYA" then "RISHAV,DEV" → first="RISHAV", full="RISHAV ACHARYA"
    if 'name' not in result:
        header_words = re.compile(
            r'^(NEW|YORK|STATE|DRIVER|LICENSE|LEARNER|PERMIT|IDENTIFICATION|FEDERAL|'
            r'PURPOSES|NONE|RESTRICTIONS|ENDORSEMENTS|CLASS|REAL|ID|NOT|FOR|UNDER)$', re.I)
        pure_caps = re.compile(r'^[A-Z]{2,25}$')
        for i, line in enumerate(lines):
            last = line.strip()
            if not pure_caps.match(last) or header_words.match(last):
                continue
            if i + 1 >= len(lines):
                break
            # First name: take part before first comma, dot, or space
            first = re.split(r'[,\.\s]+', lines[i + 1].strip())[0].strip()
            if re.match(r'^[A-Z]{2,}$', first):
                full = f'{first} {last}'
                result['name'] = full
                print(f'[PY][Regex] name from consecutive lines: '
                      f'last="{last}" first="{first}" → "{full}"')
                break

    return result


def extract_fields_regex(raw_text, lines):
    """
    Regex-based fallback that fills any fields spatial extraction missed.
    Wraps _extract_fields_regex_flat() and tags each result with source/confidence.
    Returns {fieldName: {"value": str, "confidence": 0.5, "source": "regex"}}.
    """
    flat = _extract_fields_regex_flat(raw_text, lines)
    print(f"[Regex fallback] Extracted: {list(flat.keys())}")
    return {k: {"value": v, "confidence": 0.5, "source": "regex"} for k, v in flat.items()}


def extract_fields_ollama(raw_text):
    """
    Use a local Ollama LLM as a gap-filler when regex/spatial leave key fields empty.

    Sends the raw OCR text with a structured prompt and parses the JSON response.
    Only called when name, dateOfBirth, or idNumber are still null after all other
    passes — so it never runs on clean barcoded IDs.

    Returns {fieldName: {"value": str, "confidence": 0.7, "source": "ollama"}}
    or {} on any failure (Ollama not running, bad JSON, timeout).

    Host: host.docker.internal inside Docker, localhost outside.
    Override via OLLAMA_HOST / OLLAMA_MODEL env vars.
    """
    import requests as _requests
    import json as _json

    prompt = f"""You are extracting structured fields from US ID card OCR text.
The text comes from an OCR engine and may contain noise, merged words, or missing spaces.

Return ONLY a valid JSON object with these exact keys (use null for any field not found):
  name        — full name in "FIRSTNAME LASTNAME" format
  dateOfBirth — MM/DD/YYYY
  idNumber    — ID or license number (may be all digits or letter+digits)
  address     — full street address
  expiryDate  — MM/DD/YYYY
  issueDate   — MM/DD/YYYY
  sex         — M, F, or X
  state       — 2-letter US state code

Hints:
- Name often appears as two consecutive ALL-CAPS lines: first line = LASTNAME, second = FIRSTNAME (possibly with comma or dot separating middle name). Return "FIRSTNAME LASTNAME".
- Classify dates by year: far past (>16 years ago) = dateOfBirth, future = expiryDate, recent past = issueDate.
- ID number may be space-separated digits on one line — join them.
- Ignore header words like STATE, DRIVER, LICENSE, LEARNER, PERMIT, FEDERAL, PURPOSES, NOT FOR.

OCR text:
{raw_text}"""

    try:
        resp = _requests.post(
            OLLAMA_URL,
            json={'model': OLLAMA_MODEL, 'prompt': prompt, 'format': 'json', 'stream': False},
            timeout=30,
        )
        resp.raise_for_status()
        raw_json = resp.json().get('response', '{}')
        parsed   = _json.loads(raw_json)
        print(f'[PY][Ollama] Raw response: {parsed}')
    except _requests.exceptions.ConnectionError:
        print(f'[PY][Ollama] Cannot connect to {OLLAMA_URL} — is `ollama serve` running?')
        return {}
    except Exception as e:
        print(f'[PY][Ollama] Error: {e}')
        return {}

    extracted = {}
    for key in ('name', 'dateOfBirth', 'idNumber', 'address',
                'expiryDate', 'issueDate', 'sex', 'state'):
        val = parsed.get(key)
        if val and str(val).strip().lower() not in ('null', 'none', ''):
            extracted[key] = {
                'value':      str(val).strip(),
                'confidence': 0.7,
                'source':     'ollama',
            }

    print(f'[PY][Ollama] Extracted fields: {list(extracted.keys())}')
    return extracted


def detect_document_type(mrz_data, barcode_data):
    """
    Figure out what kind of document we're looking at.
    Priority: MRZ type code → MRZ line length → barcode DCA field → 'unknown'.
    """
    if mrz_data:
        doc_type = str(mrz_data.get('type', '') or '').upper()
        if doc_type.startswith('P'):
            return 'passport'
        if doc_type.startswith(('I', 'A', 'C')):
            return 'id_card'
        if doc_type.startswith('V'):
            return 'travel_document'
        mrz_raw = mrz_data.get('raw_text', '') or ''
        lines   = [l for l in mrz_raw.split('\n') if l.strip()]
        if lines and len(lines[0]) == 44:
            return 'passport'
        if lines and len(lines[0]) == 30:
            return 'id_card'
        return 'id_card'

    if barcode_data:
        if barcode_data.get('vehicleClass') or barcode_data.get('subfile_type') == 'DL':
            return 'dl'
        return 'state_id'

    return 'unknown'


def _log_result_summary(fields, doc_type, source_tag, avg_confidence,
                        paddle_invoked, glare_ratio, mrz_data, barcode_data):
    """
    Print a formatted result summary after every OCR request.
    Shows which stages ran, per-field values with confidence and winning source.
    """
    SEP  = '─' * 62
    SEP2 = '═' * 62
    print(f'\n[PY] {SEP2}')
    print(f'[PY]  RESULT SUMMARY')
    print(f'[PY] {SEP2}')
    print(f'[PY]  Document type  : {doc_type}')
    print(f'[PY]  Source         : {source_tag}')
    print(f'[PY]  docTR avg conf : {avg_confidence:.1f}%')
    print(f'[PY]  Paddle invoked : {"YES  <-- low docTR confidence triggered it" if paddle_invoked else "no"}')
    print(f'[PY]  MRZ detected   : {"YES" if mrz_data else "no"}')
    print(f'[PY]  Barcode decoded: {"YES" if barcode_data else "no"}')
    if glare_ratio > 0.0:
        warn = '  *** HIGH — may affect accuracy' if glare_ratio > 0.15 else ''
        print(f'[PY]  Glare ratio    : {glare_ratio:.1%}{warn}')
    print(f'[PY] {SEP}')
    print(f'[PY]  {"FIELD":<14}  {"VALUE":<32}  {"CONF":>5}  SOURCE')
    print(f'[PY] {SEP}')
    for fname in ('name', 'dateOfBirth', 'address', 'idNumber',
                  'expiryDate', 'issueDate', 'sex', 'state'):
        entry   = fields.get(fname) or {}
        value   = entry.get('value') or ''
        conf    = entry.get('confidence', 0.0)
        src     = entry.get('source') or '—'
        display = (value[:29] + '...') if len(value) > 32 else value
        conf_s  = f'{conf:.3f}' if value else '  —  '
        missing = '  <-- MISSING' if not value else ''
        print(f'[PY]  {fname:<14}  {display:<32}  {conf_s:>5}  {src}{missing}')
    print(f'[PY] {SEP2}\n')


@app.route('/health', methods=['GET'])
def health():
    """
    Simple health check endpoint. Returns device info and which optional libs
    are available (including PaddleOCR).
    """
    if torch.cuda.is_available():
        device_name = "cuda"
    elif torch.backends.mps.is_available():
        device_name = "mps"
    else:
        device_name = "cpu"

    return jsonify({
        "status":       "ok",
        "ocr":          "doctr",
        "device":       device_name,
        "passporteye":  PASSPORTEYE_AVAILABLE,
        "pyzbar":       PYZBAR_AVAILABLE,
        "paddle":       PADDLE_AVAILABLE,
        "paddle_enabled": PADDLE_ENABLED,
    })


@app.route('/ocr', methods=['POST'])
def ocr():
    """
    Main OCR endpoint. Accepts an image and optional 'side' hint ('front'|'back').

    Pipeline:
      1. docTR (front/unspecified) — spatial extraction + regex gap-fill
      2. PaddleOCR PP-OCRv5 (front/unspecified, conditional)
         — triggered if name/dateOfBirth/idNumber have docTR confidence < 0.75
         — preprocessing: perspective correction → glare masking → Sauvola binarize
      3. PDF417 barcode (back/unspecified)
      4. PassportEye MRZ (front/unspecified)
      5. Merge with priority: barcode > MRZ > paddle/spatial/regex

    Each field in the response carries {value, confidence, source}.
    """
    file_bytes, err = load_bytes(request)
    if err:
        return jsonify({"error": err}), 400

    side = None
    if request.is_json:
        side = request.json.get('side')

    print(f'\n[PY] {"═" * 62}')
    print(f'[PY]  NEW REQUEST  {len(file_bytes) / 1024:.1f} KB  |  side={side or "unspecified"}')
    print(f'[PY]  Stages: docTR={"ON" if side != "back" else "skip"}  '
          f'barcode={"ON" if side != "front" else "skip"}  '
          f'mrz={"ON" if side != "back" else "skip"}  '
          f'paddle={"ON (if low conf)" if PADDLE_AVAILABLE and side != "back" else "skip"}')
    print(f'[PY] {"═" * 62}')

    img_pil   = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    img_array = np.array(img_pil)
    print(f"[docTR] Original size: {img_pil.size}")

    # ── Step 1: docTR (front or unspecified) ──────────────────────────────────
    spatial_fields = {}
    lines_out      = []
    avg_confidence = 0.0
    raw_text       = ''

    if side != 'back':
        img_preprocessed = preprocess(img_array)
        print(f"[docTR] Preprocessed size: {img_preprocessed.shape[1]}x{img_preprocessed.shape[0]}")

        _buf = io.BytesIO()
        Image.fromarray(img_preprocessed).save(_buf, format='JPEG')
        doc    = DocumentFile.from_images([_buf.getvalue()])
        result = model(doc)

        words_geo = collect_words_with_geometry(result)
        print(f"[docTR] Word count for spatial: {len(words_geo)}")
        print(f"[docTR] Words sample: {[w['text'] for w in words_geo[:30]]}")

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

    # ── Step 2: PDF417 barcode (back or unspecified) ───────────────────────────
    barcode_data = None
    if side != 'front' and PYZBAR_AVAILABLE:
        print("[Barcode] Scanning for PDF417...")
        barcode_data = decode_pdf417(img_pil)
        if barcode_data:
            print(f"[Barcode] Found: {barcode_data}")

    # ── Step 3: PassportEye MRZ (front or unspecified) ────────────────────────
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

    # ── Step 4: Document type ──────────────────────────────────────────────────
    doc_type = detect_document_type(mrz_data, barcode_data)
    print(f"[DocType] Detected: {doc_type}")

    # ── Step 5: Regex gap-filler (always runs if we have text) ────────────────
    if raw_text:
        text_lines   = [l.strip() for l in raw_text.split('\n') if l.strip()]
        print(f"[Regex] Running on {len(text_lines)} lines to fill gaps left by spatial")
        regex_fields = extract_fields_regex(raw_text, text_lines)
        for key, entry in regex_fields.items():
            if entry['value'] and not spatial_fields.get(key):
                spatial_fields[key] = entry
                print(f"[Regex] Filled '{key}' = '{entry['value']}'")

    # ── Step 5.5: Ollama gap-filler (missing key fields after regex) ──────────
    # Fires only when name/dateOfBirth/idNumber are still null — never on clean
    # barcoded IDs. If Ollama isn't running the connection error is caught silently.
    ollama_invoked = False
    if OLLAMA_ENABLED and raw_text:
        def _missing(field_name):
            entry = spatial_fields.get(field_name)
            return entry is None or not entry.get('value')

        if any(_missing(f) for f in KEY_FIELDS):
            print(f'[PY][Ollama] Triggering — key fields still empty after regex')
            ollama_fields  = extract_fields_ollama(raw_text)
            ollama_invoked = bool(ollama_fields)
            for key, entry in ollama_fields.items():
                if entry['value'] and not spatial_fields.get(key):
                    spatial_fields[key] = entry
                    print(f'[PY][Ollama] Filled "{key}" = "{entry["value"]}"')
        else:
            print(f'[PY][Ollama] Skipped — key fields already populated by earlier passes')

    # ── Step 5.9: PaddleOCR (triggered when docTR confidence is low) ──────────
    # Runs only on front-side images; skipped entirely if PADDLE_AVAILABLE=False.
    # Trigger condition: name, dateOfBirth, or idNumber are missing OR conf < 0.75.
    KEY_FIELDS     = ('name', 'dateOfBirth', 'idNumber')
    glare_ratio    = 0.0
    paddle_invoked = False

    if PADDLE_AVAILABLE and side != 'back':
        def _low_conf(field_name):
            entry = spatial_fields.get(field_name)
            return entry is None or entry.get('confidence', 0.0) < 0.75

        if any(_low_conf(f) for f in KEY_FIELDS):
            print(f"[Paddle] Triggering — low/missing confidence on key fields")
            paddle_input, glare_ratio = preprocess_for_paddle(img_array)
            print(f"[Paddle] Glare ratio: {glare_ratio:.2%}")
            paddle_fields  = run_paddle_ocr(paddle_input)
            paddle_invoked = bool(paddle_fields)
            for key, entry in paddle_fields.items():
                if entry['value'] and not spatial_fields.get(key):
                    spatial_fields[key] = entry
                    print(f"[Paddle] Filled '{key}' = '{entry['value']}'")

    # ── Step 6: Merge with priority barcode > MRZ > spatial/regex/paddle ──────

    def _field(value, confidence, source):
        return {"value": value, "confidence": confidence, "source": source}

    null_field = {"value": None, "confidence": 0.0, "source": None}
    fields = {k: null_field.copy()
              for k in ('name', 'dateOfBirth', 'address', 'idNumber',
                        'expiryDate', 'issueDate', 'sex', 'state')}

    # Spatial / regex / paddle results (lowest priority)
    for key in ('name', 'dateOfBirth', 'address', 'idNumber', 'expiryDate', 'issueDate', 'sex'):
        if spatial_fields.get(key):
            fields[key] = spatial_fields[key]

    # MRZ overrides spatial/regex — structured and reliable
    if mrz_data:
        surname     = (mrz_data.get('surname') or '').replace('<', ' ').strip()
        given_names = (mrz_data.get('given_names') or '').replace('<', ' ').strip()
        mrz_name    = ' '.join(filter(None, [given_names, surname])) or None
        mrz_dob     = convert_mrz_date(mrz_data.get('date_of_birth'))
        mrz_id      = (mrz_data.get('document_number') or '').replace('<', '').strip() or None
        mrz_state   = mrz_data.get('country') or mrz_data.get('nationality')
        if mrz_name:  fields['name']        = _field(mrz_name,  0.95, 'mrz')
        if mrz_dob:   fields['dateOfBirth'] = _field(mrz_dob,   0.95, 'mrz')
        if mrz_id:    fields['idNumber']    = _field(mrz_id,    0.95, 'mrz')
        if mrz_state: fields['state']       = _field(mrz_state, 0.95, 'mrz')

    # Barcode overrides everything — 100% structured AAMVA data, no interpretation
    if barcode_data:
        addr_parts = [
            barcode_data.get('street') or '',
            barcode_data.get('city')   or '',
            barcode_data.get('state')  or '',
            barcode_data.get('zip')    or '',
        ]
        address = ', '.join(p for p in addr_parts if p) or None
        first   = barcode_data.get('firstName')  or ''
        last    = barcode_data.get('lastName')   or ''
        middle  = barcode_data.get('middleName') or ''
        name    = ' '.join(filter(None, [first, middle, last])) or None
        bc_map  = {
            'name':        name,
            'dateOfBirth': barcode_data.get('dateOfBirth'),
            'address':     address,
            'idNumber':    barcode_data.get('idNumber'),
            'expiryDate':  barcode_data.get('expiryDate'),
            'issueDate':   barcode_data.get('issueDate'),
            'sex':         barcode_data.get('sex'),
            'state':       barcode_data.get('state'),
        }
        for key, val in bc_map.items():
            if val:
                fields[key] = _field(val, 1.0, 'barcode')

    # ── Build source tag ───────────────────────────────────────────────────────
    source_parts = []
    if side != 'back':
        source_parts.append('doctr')
        if ollama_invoked:
            source_parts.append('ollama')
        if paddle_invoked:
            source_parts.append('paddle')
        if mrz_data:
            source_parts.append('passporteye')
    if barcode_data:
        source_parts.append('barcode')
    source_tag = '+'.join(source_parts) if source_parts else 'doctr'

    # ── Warnings ───────────────────────────────────────────────────────────────
    warnings = {}
    if glare_ratio > 0.15:
        warnings['glare_ratio'] = round(glare_ratio, 4)

    response = {
        "success":      True,
        "documentType": doc_type,
        "side":         side,
        "fields":       fields,
        "confidence":   round(avg_confidence, 2),
        "source":       source_tag,
        "raw_text":     raw_text,
        "lines":        lines_out,
        "mrz":          mrz_data,
        "barcode":      barcode_data,
    }
    if warnings:
        response["warnings"] = warnings

    _log_result_summary(fields, doc_type, source_tag, avg_confidence,
                        paddle_invoked, glare_ratio, mrz_data, barcode_data)

    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3002, debug=False)
