FROM python:3.11-slim

# System deps:
#   tesseract-ocr + eng data  → PassportEye
#   libgl1 + libglib2.0-0     → OpenCV (used internally by EasyOCR)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download EasyOCR English model so it's baked into the image.
# This avoids a 100 MB download on every cold start.
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"

COPY app.py .

EXPOSE 3002

CMD ["python", "app.py"]
