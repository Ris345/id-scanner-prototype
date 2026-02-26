import io
import base64
import easyocr
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

try:
    from passporteye import read_mrz
    PASSPORTEYE_AVAILABLE = True
except ImportError:
    PASSPORTEYE_AVAILABLE = False
    print("PassportEye not available — install passporteye + tesseract")

app = Flask(__name__)

print("Loading EasyOCR model...")
reader = easyocr.Reader(['en'], gpu=False)
print("EasyOCR ready.")


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


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "easyocr": True,
        "passporteye": PASSPORTEYE_AVAILABLE,
    })


@app.route('/ocr', methods=['POST'])
def ocr():
    file_bytes, err = load_bytes(request)
    if err:
        return jsonify({"error": err}), 400

    print(f"\n[OCR] Received image: {len(file_bytes) / 1024:.1f} KB")

    # ── EasyOCR: full-image text extraction ──────────────────────────────────
    img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    img_array = np.array(img)
    print(f"[EasyOCR] Running on image size: {img.size}")

    results = reader.readtext(img_array)

    lines = []
    total_confidence = 0.0
    for (_, text, confidence) in results:
        text = text.strip()
        if text:
            lines.append({"text": text, "confidence": round(confidence, 4)})
            total_confidence += confidence

    avg_confidence = (total_confidence / len(lines) * 100) if lines else 0.0
    raw_text = "\n".join(item["text"] for item in lines)

    print(f"[EasyOCR] Found {len(lines)} text regions | avg confidence: {avg_confidence:.1f}%")
    print(f"[EasyOCR] Raw text:\n{raw_text}\n")

    # ── PassportEye: MRZ zone detection + structured field extraction ─────────
    mrz_data = None
    if PASSPORTEYE_AVAILABLE:
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
    else:
        print("[PassportEye] Skipped (not installed)")

    print(f"[OCR] Done — returning {'MRZ data' if mrz_data else 'EasyOCR text only'}\n")

    return jsonify({
        "success": True,
        "raw_text": raw_text,
        "lines": lines,
        "confidence": round(avg_confidence, 2),
        "mrz": mrz_data,
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3002, debug=False)
