'use strict';

const express = require('express');
const multer = require('multer');
const cors = require('cors');
const path = require('path');
try { require('dotenv').config({ path: path.join(__dirname, '../.env') }); } catch {}

const app = express();
const PORT = process.env.PORT || 3001;
const PYTHON_OCR_URL = process.env.PYTHON_OCR_URL || 'http://localhost:3002';

app.use(cors());
app.use(express.json({ limit: '50mb' }));

const storage = multer.memoryStorage();
const upload = multer({ storage, limits: { fileSize: 50 * 1024 * 1024 } });

// ── State abbreviation → full name ────────────────────────────────────────────
const STATE_MAP = {
  'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
  'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
  'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
  'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
  'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
  'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
  'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
  'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
  'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
  'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming',
};

// ── Textract (optional failsafe — only used if Python service is down) ─────────
let textractClient = null;

function textractAvailable() {
  return !!(process.env.AWS_ACCESS_KEY_ID && process.env.AWS_SECRET_ACCESS_KEY);
}

function initTextract() {
  if (!textractAvailable()) {
    console.warn('AWS credentials not set — Textract unavailable');
    return;
  }
  const { TextractClient } = require('@aws-sdk/client-textract');
  textractClient = new TextractClient({
    region: process.env.AWS_REGION || 'us-east-1',
    credentials: {
      accessKeyId: process.env.AWS_ACCESS_KEY_ID,
      secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
    },
  });
  console.log('Textract client initialized (region:', process.env.AWS_REGION || 'us-east-1', ')');
}

async function scanWithTextract(imageBuffer) {
  if (!textractClient) throw new Error('Textract client not initialized');
  const { AnalyzeIDCommand } = require('@aws-sdk/client-textract');
  console.log('[Textract] Sending to AnalyzeID...');
  const result = await textractClient.send(new AnalyzeIDCommand({
    DocumentPages: [{ Bytes: imageBuffer }],
  }));
  const fields = {};
  const rawParts = [];
  for (const doc of result.IdentityDocuments || []) {
    for (const field of doc.IdentityDocumentFields || []) {
      const key = field.Type?.Text;
      const value = field.ValueDetection?.Text;
      const confidence = field.ValueDetection?.Confidence;
      if (key && value) {
        fields[key] = value;
        rawParts.push(`${key}: ${value} (${confidence?.toFixed(1)}%)`);
      }
    }
  }
  const first = fields['FIRST_NAME'] || '';
  const middle = fields['MIDDLE_NAME'] || '';
  const last = fields['LAST_NAME'] || fields['SURNAME'] || '';
  const street = fields['ADDRESS'] || '';
  const city = fields['CITY_IN_ADDRESS'] || '';
  const state = fields['STATE_IN_ADDRESS'] || '';
  const zip = fields['ZIP_CODE_IN_ADDRESS'] || '';
  const addrParts = [street, city, state].filter(Boolean);
  const addr = (zip ? `${addrParts.join(', ')} ${zip}` : addrParts.join(', ')) || null;
  return {
    data: {
      name: [first, middle, last].filter(Boolean).join(' ') || fields['NAME'] || null,
      dateOfBirth: fields['DATE_OF_BIRTH'] || null,
      address: fields['ADDRESS'] || addr,
      idNumber: fields['DOCUMENT_NUMBER'] || fields['ID_NUMBER'] || null,
      expiryDate: null,
      issueDate: null,
      sex: null,
      state: fields['STATE_NAME'] || fields['STATE'] || null,
      documentType: null,
    },
    rawText: rawParts.join('\n'),
  };
}

// ── MRZ date (YYMMDD → MM/DD/YYYY) ────────────────────────────────────────────
function convertMrzDate(yymmdd) {
  if (!yymmdd || !/^\d{6}$/.test(String(yymmdd))) return null;
  const s = String(yymmdd);
  const year = parseInt(s.substring(0, 2), 10);
  const fullYear = year > 30 ? 1900 + year : 2000 + year;
  return `${s.substring(2, 4)}/${s.substring(4, 6)}/${fullYear}`;
}

// ── Python microservice call ───────────────────────────────────────────────────
async function scanWithPython(imageBuffer, side) {
  const body = { image: imageBuffer.toString('base64') };
  if (side) body.side = side;

  const res = await fetch(`${PYTHON_OCR_URL}/ocr`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(30000),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Python OCR service error ${res.status}: ${text}`);
  }

  const ocrResult = await res.json();
  if (!ocrResult.success) throw new Error(`Python OCR failed: ${ocrResult.error || 'unknown'}`);

  const rawText = ocrResult.raw_text || '';
  const confidence = ocrResult.confidence || 0;
  console.log(`[OCR] Python confidence: ${confidence.toFixed ? confidence.toFixed(1) : confidence}%`);
  console.log(`[OCR] Raw text:\n${rawText}`);

  // Python now returns structured fields directly
  const f = ocrResult.fields || {};

  // Expand 2-letter state abbreviation → full name (barcode returns abbreviations)
  const stateRaw = f.state || null;
  const state = (stateRaw && STATE_MAP[stateRaw.toUpperCase()]) ? STATE_MAP[stateRaw.toUpperCase()] : stateRaw;

  const data = {
    name:         f.name         || null,
    dateOfBirth:  f.dateOfBirth  || null,
    address:      f.address      || null,
    idNumber:     f.idNumber     || null,
    expiryDate:   f.expiryDate   || null,
    issueDate:    f.issueDate    || null,
    sex:          f.sex          || null,
    state:        state,
    documentType: ocrResult.documentType || null,
  };

  console.log('[OCR] Parsed data:', JSON.stringify(data, null, 2));
  return { data, rawText };
}

// ── Routes ────────────────────────────────────────────────────────────────────
app.get('/', async (req, res) => {
  let pythonAlive = false;
  try {
    const r = await fetch(`${PYTHON_OCR_URL}/health`, { signal: AbortSignal.timeout(3000) });
    pythonAlive = r.ok;
  } catch {}
  res.json({
    status: 'ID Scanner API running',
    version: '7.0.0',
    ocr: 'docTR+PassportEye+PDF417 (primary) → Textract (failsafe if Python is down)',
    python_ocr: pythonAlive ? 'available' : 'unavailable',
    textract: textractAvailable() ? 'available' : 'unavailable',
  });
});

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

    const side = req.body.side || null;
    console.log('Received image:', (imageBuffer.length / 1024).toFixed(1), 'KB', side ? `| side=${side}` : '');

    // Step 1: Python microservice (docTR + PassportEye + PDF417) — always primary
    try {
      const result = await scanWithPython(imageBuffer, side);
      return res.json({ success: true, ...result, source: 'doctr+passporteye+barcode' });
    } catch (e) {
      console.warn('[server] Python OCR service failed:', e.message);
    }

    // Step 2: Textract — only if Python service itself is down and AWS creds are set
    if (textractAvailable() && textractClient) {
      try {
        const result = await scanWithTextract(imageBuffer);
        return res.json({ success: true, ...result, source: 'textract' });
      } catch (e) {
        console.error('[server] Textract also failed:', e.message);
      }
    }

    res.status(500).json({ error: 'All OCR methods failed' });

  } catch (error) {
    console.error('Unhandled error in /api/scan:', error);
    res.status(500).json({ error: 'Failed to process image', message: error.message });
  }
});

async function start() {
  initTextract();
  app.listen(PORT, '0.0.0.0', () => {
    console.log(`
ID Scanner Backend
━━━━━━━━━━━━━━━━━━━━━━
Server:   http://localhost:${PORT}
OCR:      docTR+PassportEye+PDF417 (primary) → Textract (failsafe if Python is down)
Python:   ${PYTHON_OCR_URL}
Textract: ${textractAvailable() ? 'available' : 'unavailable (no AWS creds)'}

Endpoints:
  GET  /           Health check
  POST /api/scan   Scan ID image  (body: { image, side?: 'front'|'back' })
`);
  });
}

start().catch(err => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
