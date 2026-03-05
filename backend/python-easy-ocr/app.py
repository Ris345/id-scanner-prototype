import io
import re
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

import torch
from doctr.models import ocr_predictor
from doctr.io import DocumentFile

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
    """
    Pull the raw image bytes out of the incoming HTTP request.
    Supports two ways the caller can send an image:
      1. Multipart form upload (req.files['image']) — standard file upload
      2. JSON body with base64-encoded string (req.json['image']) — what the app sends
    Returns (bytes, None) on success or (None, error_message) on failure.
    """
    if 'image' in req.files:
        # Multipart upload — just read the file bytes directly
        return req.files['image'].read(), None
    elif req.is_json and req.json.get('image'):
        b64 = req.json['image']
        # Strip the "data:image/jpeg;base64," prefix if present
        if ',' in b64:
            b64 = b64.split(',', 1)[1]
        return base64.b64decode(b64), None
    return None, "No image provided"


def preprocess(img_array):
    """
    Prepare an ID card photo for OCR. Two things happen here:

    1. Upscale to at least 1200px wide.
       docTR needs enough pixels to detect small text. Most phone photos are
       fine, but low-res uploads would fail without this.

    2. CLAHE (Contrast Limited Adaptive Histogram Equalization).
       This boosts local contrast to handle glare, shadows, and washed-out areas
       common in ID photos. We apply it only to the luminance (L) channel in LAB
       color space so colors don't shift.
    """
    h, w = img_array.shape[:2]

    # Only upscale if the image is narrower than 1200px
    if w < 1200:
        scale = 1200 / w
        img_array = cv2.resize(
            img_array,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC,  # best quality upscaling algorithm
        )

    # Convert to LAB color space so we can touch only the lightness channel
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L (lightness) — clipLimit caps how aggressively it boosts contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # Put channels back together and convert back to RGB
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


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
        # AAMVA data is latin-1 encoded, not UTF-8
        text = raw_bytes.decode('latin-1')
    except Exception:
        return None

    # Split into lines and extract 3-letter code + value pairs
    lines = re.split(r'[\r\n]+', text)
    fields = {}
    for line in lines:
        m = re.match(r'^([A-Z]{3})(.+)$', line.strip())
        if m:
            code, value = m.group(1), m.group(2).strip()
            fields[code] = value

    def fmt_date(d):
        """AAMVA dates are MMDDYYYY — reformat to MM/DD/YYYY."""
        if d and re.match(r'^\d{8}$', d):
            return f"{d[0:2]}/{d[2:4]}/{d[4:8]}"
        return d

    sex_map = {'1': 'M', '2': 'F', '9': 'X'}

    # If DCA (vehicle class) is present it's a driver's license, otherwise state ID
    subfile_type = 'DL' if 'DCA' in fields else 'ID'

    first = fields.get('DCT') or fields.get('DAC')
    last = fields.get('DCS')

    # Some states cram "SMITH,JOHN" into the first-name field — split it out
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
        'state':        fields.get('DAJ'),   # 2-letter abbreviation
        'zip':          fields.get('DAK'),
        'idNumber':     fields.get('DAQ'),
        'sex':          sex_map.get(fields.get('DBC', ''), fields.get('DBC')),
        'vehicleClass': fields.get('DCA'),   # e.g. "C" for regular car license
        'subfile_type': subfile_type,
    }
    return result


def decode_pdf417(img_pil):
    """
    Try to decode the PDF417 barcode from an ID card back image.
    PDF417 is the tall, multi-row 2D barcode on the back of US driver's licenses.

    We try three strategies in order because phone cameras and cropping can make
    the barcode tricky to read:
      1. Full preprocessed image — works most of the time
      2. Bottom-half crop — barcode is usually in the lower half of the card
      3. 2× upscale — helps when the image is low resolution

    Returns the parsed AAMVA dict, or None if nothing decoded.
    """
    if not PYZBAR_AVAILABLE:
        return None

    img_array = np.array(img_pil.convert('RGB'))

    def try_decode(arr):
        # pyzbar works on grayscale images; explicitly tell it to only look for PDF417
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        codes = pyzbar_decode(gray, symbols=[ZBarSymbol.PDF417])
        if codes:
            return parse_aamva_barcode(codes[0].data)
        return None

    # Strategy 1: full image
    result = try_decode(img_array)
    if result:
        print("[Barcode] Decoded from full image")
        return result

    # Strategy 2: bottom half only (cuts out header graphics that confuse the decoder)
    h = img_array.shape[0]
    result = try_decode(img_array[h // 2:, :])
    if result:
        print("[Barcode] Decoded from bottom-half crop")
        return result

    # Strategy 3: 2× upscale (helps with blurry or compressed images)
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

    docTR organizes results as: page → block → line → word.
    We flatten that into a simple list of dicts, each with:
      text  — the recognized string
      conf  — confidence 0.0–1.0
      x_min, y_min, x_max, y_max — corners of the word's bounding box
      cx, cy — center point of the box

    All coordinates are normalized (0.0 = left/top edge, 1.0 = right/bottom edge)
    so they work regardless of image size.
    """
    words_out = []
    for page in doctr_result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    # docTR geometry format: ((x_min, y_min), (x_max, y_max))
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
    Search the word list for a label word (or multi-word label phrase).

    Example: alias "DATE OF BIRTH" requires three consecutive words that spell
    exactly "DATE OF BIRTH". We return the LAST word of the matching span because
    the value we want is to the right of or below that last word.

    Returns the matching word dict, or None if no alias was found.
    """
    for alias in aliases:
        alias_tokens = alias.upper().split()  # e.g. ["DATE", "OF", "BIRTH"]
        n = len(alias_tokens)
        # Slide a window of size n across the word list
        for i in range(len(words) - n + 1):
            span = [words[i + j]['text'].upper().rstrip(':') for j in range(n)]
            if span == alias_tokens:
                return words[i + n - 1]  # return last word of the matched span
    return None


def get_value_words_to_right(words, label_word):
    """
    Find words that are to the RIGHT of the label on the same row.

    'Same row' means the word's vertical center (cy) is within SAME_ROW_Y_TOLERANCE
    of the label's center — this absorbs small vertical jitter from OCR.

    We require x_min > label's x_max (minus a tiny overlap buffer) so we only grab
    words that actually come after the label, not the label itself.

    Returns up to 6 words sorted left-to-right.
    """
    label_x_max = label_word['x_max']
    label_cy    = label_word['cy']
    candidates = [
        w for w in words
        if w['x_min'] > label_x_max - 0.01              # to the right of label
        and abs(w['cy'] - label_cy) < SAME_ROW_Y_TOLERANCE  # same row
        and w is not label_word
    ]
    candidates.sort(key=lambda w: w['x_min'])
    return candidates[:6]


def get_value_words_below(words, label_word):
    """
    Find words that are BELOW the label and roughly in the same horizontal column.

    This handles the common DL layout where the label is on one line and the
    value is on the next line directly underneath it.

    'Below' means y_min > label's y_max (minus a tiny buffer).
    'Same column' means the word's x range overlaps the label's x range
    (within SAME_COL_X_TOLERANCE padding on each side).

    Returns up to 2 rows, 8 words each.
    """
    label_y_max = label_word['y_max']
    label_x_min = label_word['x_min']
    label_x_max = label_word['x_max']
    candidates = [
        w for w in words
        if w['y_min'] > label_y_max - 0.01                          # below label
        and w['x_min'] < label_x_max + SAME_COL_X_TOLERANCE         # x-overlap check
        and w['x_max'] > label_x_min - SAME_COL_X_TOLERANCE
    ]
    if not candidates:
        return []

    # Group candidates into rows by their vertical center proximity
    candidates.sort(key=lambda w: w['cy'])
    rows = []
    current_row = []
    current_y   = None
    for w in candidates:
        if current_y is None or abs(w['cy'] - current_y) < SAME_ROW_Y_TOLERANCE:
            current_row.append(w)
            # Running average of row's y center
            current_y = w['cy'] if current_y is None else (current_y + w['cy']) / 2
        else:
            rows.append(current_row)
            current_row = [w]
            current_y   = w['cy']
            if len(rows) >= 2:
                break  # we only want the first 2 rows below the label
    if current_row and len(rows) < 2:
        rows.append(current_row)

    result = []
    for row in rows[:2]:
        result.extend(row[:8])
    return result


def extract_fields_spatially(words):
    """
    The main spatial extraction pass.

    For each field we want (name, DOB, address, etc.) we:
      1. Look through the word list for a label word matching any known alias
      2. Try to grab value words to the RIGHT of that label first (same-line layout)
      3. Fall back to words BELOW the label (next-line layout)
      4. Join the found words into a string and store it

    If first/last name were found separately but 'name' wasn't, we merge them.

    Returns a dict of {fieldName: value} for whatever fields were found.
    """
    extracted = {}
    for field, aliases in LABEL_ALIASES.items():
        label_word = find_label_word(words, aliases)
        if not label_word:
            continue  # this label doesn't appear on the card

        # Try right-of-label first (most common layout), then below
        value_words = get_value_words_to_right(words, label_word)
        if not value_words:
            value_words = get_value_words_below(words, label_word)

        if value_words:
            value = ' '.join(w['text'] for w in value_words).strip()
            if value:
                extracted[field] = value

    # Merge first + last into a single 'name' field if we didn't find 'name' directly
    if 'name' not in extracted and ('firstName' in extracted or 'lastName' in extracted):
        parts  = [extracted.get('firstName', ''), extracted.get('lastName', '')]
        merged = ' '.join(p for p in parts if p).strip()
        if merged:
            extracted['name'] = merged

    print(f"[Spatial] Extracted: {extracted}")
    return extracted


def extract_fields_regex(raw_text, lines):
    """
    Regex-based fallback that fills any fields spatial extraction missed.

    Why do we need this alongside spatial?
    Spatial extraction needs docTR to correctly detect a bounding box around the
    label word. If it misses one (low contrast, small font, partial crop), spatial
    gets nothing. Regex works directly on the flat text string so it's a safety net.

    Handles TWO common DL label layouts:
      - Same-line:  "LN SMITH"   (label and value on the same line)
      - Next-line:  "LN\nSMITH" (label on line i, value on line i+1)

    The inner helper val_same_or_next() handles both cases automatically.

    After label-based extraction, there's also a label-FREE pattern pass that
    looks for things like "A1234567" (letter + digits = ID number) and groups dates
    by year to guess which is DOB vs expiry vs issue date.
    """
    result = {}

    def val_same_or_next(i, strip_pat):
        """
        Look for a value on line i after stripping the label prefix.
        If nothing is left on line i, check line i+1 (next-line layout).
        """
        after = re.sub(strip_pat, '', lines[i], flags=re.I).strip()
        if after:
            return after
        if i + 1 < len(lines):
            return lines[i + 1].strip()
        return ''

    # ── Date of birth ─────────────────────────────────────────────────────────
    # Look for a DOB label first; if not found, grab the first date in the text
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

    # ── Expiry date ───────────────────────────────────────────────────────────
    exp_label = re.compile(r'\b(EXP|EXPIRES|EXPIRY|EXPIRATION)\b', re.I)
    for i, line in enumerate(lines):
        if exp_label.search(line):
            candidate = val_same_or_next(i, r'^.*?(EXP|EXPIRES|EXPIRY|EXPIRATION)[:\s]*')
            m = date_pat.search(candidate)
            if m:
                result['expiryDate'] = m.group(0)
                break

    # ── Issue date ────────────────────────────────────────────────────────────
    iss_label = re.compile(r'\b(ISS|ISSUED|ISSUE\s+DATE)\b', re.I)
    for i, line in enumerate(lines):
        if iss_label.search(line):
            candidate = val_same_or_next(i, r'^.*?(ISS|ISSUED|ISSUE\s+DATE)[:\s]*')
            m = date_pat.search(candidate)
            if m:
                result['issueDate'] = m.group(0)
                break

    # ── ID / licence number ───────────────────────────────────────────────────
    # Try labeled extraction first; fall back to "letter followed by 7+ digits"
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

    # ── Name ──────────────────────────────────────────────────────────────────
    # Try labeled extraction (NAME / FIRST / LAST labels), then fall back to
    # finding the first line that looks like a name (2–4 pure-alpha words, no header)
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
        if re.search(r'\b(FIRST|GIVEN|FN)\b', line, re.I):
            val = val_same_or_next(i, r'^.*?(FIRST|GIVEN|FN)[:\s]*')
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
            # Last resort: first line with 2–4 pure-alpha words that isn't a header
            for line in lines:
                if re.search(r'\d', line) or len(line) < 4:
                    continue
                if header_skip.match(line.strip()):
                    continue
                words = [w for w in line.split() if re.match(r'^[A-Za-z]+$', w) and len(w) > 1]
                if 2 <= len(words) <= 4:
                    result['name'] = ' '.join(words)
                    break

    # ── Address ───────────────────────────────────────────────────────────────
    # Start capturing when we see "ADDRESS" label, continue until we hit a ZIP code
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
                break  # ZIP signals the end of the address block
    if addr_lines:
        result['address'] = ', '.join(addr_lines)

    # ── Sex ───────────────────────────────────────────────────────────────────
    for i, line in enumerate(lines):
        if re.search(r'\bSEX\b', line, re.I):
            candidate = val_same_or_next(i, r'^.*?SEX[:\s]*')
            m = re.search(r'\b(MALE|FEMALE|[MFX])\b', candidate, re.I)
            if m:
                v = m.group(0).upper()
                # Normalize to single letter
                result['sex'] = 'M' if v == 'MALE' else 'F' if v == 'FEMALE' else v
                break

    # ── Label-free pattern fallback ───────────────────────────────────────────
    # For fields still missing, try to find values by pattern alone — no label needed.

    # ID number: any token matching letter + 6–9 digits (covers all US state formats)
    if 'idNumber' not in result:
        for line in lines:
            m = re.search(r'\b([A-Z]\d{6,9})\b', line)
            if m:
                result['idNumber'] = m.group(1)
                break

    # Dates: collect all date-shaped tokens and classify them by year:
    #   past year (>16 years ago)  → likely DOB
    #   future or current year     → likely expiry
    #   recent past (0–10 years)   → likely issue date
    if 'dateOfBirth' not in result or 'expiryDate' not in result or 'issueDate' not in result:
        import datetime
        current_year = datetime.datetime.now().year
        all_dates = re.findall(r'\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})\b', raw_text)
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

    # Name: last resort — skip header/state-name lines, grab first 2–5 word alpha line
    if 'name' not in result:
        skip = re.compile(
            r'^(CALIFORNIA|FLORIDA|TEXAS|NEW\s+YORK|DRIVER|LICENSE|LICENCE|IDENTIFICATION|'
            r'ID\s+CARD|CLASS|NOT\s+FOR|FEDERAL|REAL\s+ID|DOB|SEX|EXP|ISS|HT|WT|DL|DLN|'
            r'EYES?|HAIR|RESTRICTIONS|ENDORSEMENTS|VETERAN|DONOR|ORGAN|NONE|UNDER|'
            r'[A-Z]{2}\s+\d|DD\s).*$', re.I)
        for line in lines:
            if re.search(r'\d', line):
                continue
            if skip.match(line.strip()):
                continue
            words = [w for w in line.split() if re.match(r'^[A-Za-z\-\']+$', w) and len(w) > 1]
            if 2 <= len(words) <= 5:
                result['name'] = ' '.join(words)
                break

    print(f"[Regex fallback] Extracted: {result}")
    return result


def detect_document_type(mrz_data, barcode_data):
    """
    Figure out what kind of document we're looking at.

    Priority:
      1. MRZ present → use the document type code in the MRZ
           P* = passport, I*/A*/C* = ID card, V* = travel/visa document
           If the type code is ambiguous, fall back to line length (44=passport, 30=ID card)
      2. Barcode present → driver's license (DL) or state ID
           DCA field in barcode = vehicle class → it's a DL
           No DCA = state_id
      3. Neither → 'unknown'
    """
    if mrz_data:
        doc_type = str(mrz_data.get('type', '') or '').upper()
        if doc_type.startswith('P'):
            return 'passport'
        if doc_type.startswith(('I', 'A', 'C')):
            return 'id_card'
        if doc_type.startswith('V'):
            return 'travel_document'
        # Fallback: check MRZ line length (passports use 44-char lines, ID cards use 30)
        mrz_raw = mrz_data.get('raw_text', '') or ''
        lines = [l for l in mrz_raw.split('\n') if l.strip()]
        if lines and len(lines[0]) == 44:
            return 'passport'
        if lines and len(lines[0]) == 30:
            return 'id_card'
        return 'id_card'  # default for any MRZ document we can't classify further

    if barcode_data:
        # DCA = vehicle class code (only DLs have this), so its presence means DL
        if barcode_data.get('vehicleClass') or barcode_data.get('subfile_type') == 'DL':
            return 'dl'
        return 'state_id'

    return 'unknown'


@app.route('/health', methods=['GET'])
def health():
    """
    Simple health check endpoint. The Node.js server pings this on startup and
    in the status route to know if the Python service is alive.
    Returns what device docTR is running on plus which optional libs are available.
    """
    if torch.cuda.is_available():
        device_name = "cuda"
    elif torch.backends.mps.is_available():
        device_name = "mps"
    else:
        device_name = "cpu"

    return jsonify({
        "status": "ok",
        "ocr": "doctr",
        "device": device_name,
        "passporteye": PASSPORTEYE_AVAILABLE,
        "pyzbar": PYZBAR_AVAILABLE,
    })


@app.route('/ocr', methods=['POST'])
def ocr():
    """
    Main OCR endpoint. Accepts an image and an optional 'side' hint ('front'|'back').

    Pipeline (in order):
      1. docTR (front/unspecified only)
         - Preprocess image (upscale + CLAHE)
         - Run OCR → get words with bounding boxes
         - Spatial extraction: find label words → grab adjacent value words
         - Regex gap-filler: fill any fields spatial missed

      2. PDF417 barcode (back/unspecified only)
         - If decoded, its fields will override spatial/regex results
         - This is the most reliable source — perfectly structured AAMVA data

      3. PassportEye MRZ (front/unspecified only)
         - Detects and reads the machine-readable zone on passports/ID cards
         - MRZ fields override spatial/regex but are overridden by barcode

      4. Merge with priority: barcode > MRZ > spatial/regex

    The 'side' hint is an optimization — skip irrelevant processing:
      side='front'  → skip barcode scan
      side='back'   → skip docTR OCR and MRZ scan
      side=None     → run everything
    """
    file_bytes, err = load_bytes(request)
    if err:
        return jsonify({"error": err}), 400

    side = None
    if request.is_json:
        side = request.json.get('side')  # 'front' | 'back' | None

    print(f"\n[OCR] Received image: {len(file_bytes) / 1024:.1f} KB | side={side}")

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

        # DocumentFile wraps the numpy array into the format docTR expects
        doc    = DocumentFile.from_images([img_preprocessed])
        result = model(doc)

        # Collect every word with its position on the page
        words_geo = collect_words_with_geometry(result)
        print(f"[docTR] Word count for spatial: {len(words_geo)}")
        print(f"[docTR] Words sample: {[w['text'] for w in words_geo[:30]]}")

        # Try to extract fields using label-proximity spatial logic
        spatial_fields = extract_fields_spatially(words_geo)

        # Also build the flat text + confidence for the regex pass and response
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
    # Fills any field that spatial extraction missed. Runs unconditionally so even
    # if spatial found a few fields, regex still fills the rest.
    if raw_text:
        text_lines   = [l.strip() for l in raw_text.split('\n') if l.strip()]
        print(f"[Regex] Running on {len(text_lines)} lines to fill gaps left by spatial")
        print(f"[Regex] Lines: {text_lines}")
        regex_fields = extract_fields_regex(raw_text, text_lines)
        for key, val in regex_fields.items():
            if val and not spatial_fields.get(key):
                spatial_fields[key] = val
                print(f"[Regex] Filled '{key}' = '{val}'")

    # ── Step 6: Merge with priority barcode > MRZ > spatial/regex ─────────────
    fields = {
        'name': None, 'dateOfBirth': None, 'address': None,
        'idNumber': None, 'expiryDate': None, 'issueDate': None,
        'sex': None, 'state': None,
    }

    # Spatial/regex fields go in first (lowest priority)
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

    # MRZ overrides spatial/regex — it's more reliable than OCR text parsing
    if mrz_data:
        surname     = (mrz_data.get('surname') or '').replace('<', ' ').strip()
        given_names = (mrz_data.get('given_names') or '').replace('<', ' ').strip()
        fields.update({
            'name':        ' '.join(filter(None, [given_names, surname])) or None,
            'dateOfBirth': convert_mrz_date(mrz_data.get('date_of_birth')),
            'idNumber':    (mrz_data.get('document_number') or '').replace('<', '').strip() or None,
            'state':       mrz_data.get('country') or mrz_data.get('nationality'),
        })

    # Barcode overrides everything — 100% structured, no interpretation needed
    if barcode_data:
        addr_parts = [
            barcode_data.get('street') or '',
            barcode_data.get('city') or '',
            barcode_data.get('state') or '',
            barcode_data.get('zip') or '',
        ]
        address = ', '.join(p for p in addr_parts if p) or None
        first   = barcode_data.get('firstName') or ''
        last    = barcode_data.get('lastName') or ''
        middle  = barcode_data.get('middleName') or ''
        name    = ' '.join(filter(None, [first, middle, last])) or None
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
        "success":      True,
        "documentType": doc_type,
        "side":         side,
        "fields":       fields,
        "confidence":   round(avg_confidence, 2),
        "raw_text":     raw_text,
        "lines":        lines_out,
        "mrz":          mrz_data,
        "barcode":      barcode_data,
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3002, debug=False)
