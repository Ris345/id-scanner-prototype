'use strict';

const express = require('express');
const multer = require('multer');
const cors = require('cors');
const sharp = require('sharp');
const path = require('path');
try { require('dotenv').config({ path: path.join(__dirname, '../.env') }); } catch {}

const { initWorker, scanWithTesseract } = require('./services/tesseractService');
const { initTextract, scanWithTextract, isAvailable } = require('./services/textractService');

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(express.json({ limit: '50mb' }));

const storage = multer.memoryStorage();
const upload = multer({ storage, limits: { fileSize: 50 * 1024 * 1024 } });

// Health check
app.get('/', (req, res) => {
  res.json({
    status: 'ID Scanner API running',
    version: '4.0.0',
    ocr: 'Tesseract+MRZ → Textract (failsafe)',
    textract: isAvailable() ? 'available' : 'unavailable',
  });
});

// Main endpoint: Parse ID from image
app.post('/api/scan', upload.single('image'), async (req, res) => {
  try {
    let imageBuffer;

    if (req.file) {
      imageBuffer = req.file.buffer;
    } else if (req.body.image) {
      const base64Data = req.body.image.replace(/^data:image\/\w+;base64,/, '');
      imageBuffer = Buffer.from(base64Data, 'base64');
    } else {
      return res.status(400).json({ error: 'No image provided' });
    }

    console.log('Received image:', (imageBuffer.length / 1024).toFixed(1), 'KB');

    // Preprocess image for better OCR accuracy
    // Pipeline: greyscale → upscale → denoise → normalize → binarize
    // threshold(128) gives Tesseract crisp black/white text, which it handles
    // much better than anti-aliased grayscale. median(3) removes speckle noise
    // before binarization so thin characters don't break up.
    const processed = await sharp(imageBuffer)
      .greyscale()
      .resize({ width: 2000, withoutEnlargement: true })
      .median(3)
      .normalize()
      .threshold(128)
      .png()
      .toBuffer();

    console.log('Preprocessed:', (processed.length / 1024).toFixed(1), 'KB');

    // Route 1: Tesseract + MRZ (primary, free)
    try {
      const result = await scanWithTesseract(processed);
      return res.json({ success: true, ...result, source: 'tesseract+mrz' });
    } catch (e) {
      console.warn('Tesseract failed, trying Textract:', e.message);
    }

    // Route 2: Textract (failsafe, costs money per call)
    if (isAvailable()) {
      try {
        const result = await scanWithTextract(processed);
        return res.json({ success: true, ...result, source: 'textract' });
      } catch (e) {
        console.error('Textract also failed:', e.message);
      }
    }

    res.status(500).json({ error: 'All OCR methods failed' });

  } catch (error) {
    console.error('Unhandled error in /api/scan:', error);
    res.status(500).json({ error: 'Failed to process image', message: error.message });
  }
});

async function start() {
  await initWorker();
  initTextract();

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`
ID Scanner Backend
━━━━━━━━━━━━━━━━━━━━━━
Server:   http://localhost:${PORT}
OCR:      Tesseract+MRZ (primary) → Textract (failsafe)
Textract: ${isAvailable() ? 'available' : 'unavailable (no AWS creds)'}

Endpoints:
  GET  /           Health check
  POST /api/scan   Scan ID image
`);
  });
}

start().catch(err => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
