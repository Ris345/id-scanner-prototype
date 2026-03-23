FROM python:3.11-slim

# System deps:
#   tesseract-ocr + eng data  → PassportEye MRZ
#   libgl1 + libglib2.0-0     → OpenCV
#   libzbar0                  → pyzbar PDF417 barcode
#   libgomp1                  → PaddlePaddle (OpenMP threading)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    libzbar0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-bake docTR models so they are never downloaded at runtime.
RUN python -c "from doctr.models import ocr_predictor; ocr_predictor(det_arch='db_resnet50', reco_arch='parseq', pretrained=True)"

COPY app.py .

EXPOSE 3002

CMD ["python", "app.py"]
