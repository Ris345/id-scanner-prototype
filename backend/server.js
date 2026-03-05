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

// ── Textract ──────────────────────────────────────────────────────────────────
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

  const f = {};
  const rawParts = [];
  for (const doc of result.IdentityDocuments || []) {
    for (const field of doc.IdentityDocumentFields || []) {
      const key   = field.Type?.Text;
      const value = field.ValueDetection?.Text;
      const conf  = field.ValueDetection?.Confidence;
      if (key && value) {
        f[key] = value;
        rawParts.push(`${key}: ${value} (${conf?.toFixed(1)}%)`);
      }
    }
  }

  console.log('[Textract] Raw fields:', f);

  const first  = f['FIRST_NAME']  || '';
  const middle = f['MIDDLE_NAME'] || '';
  const last   = f['LAST_NAME']   || f['SURNAME'] || '';
  const street = f['ADDRESS']     || f['ADDRESS_LINE_1'] || '';
  const city   = f['CITY_IN_ADDRESS'] || '';
  const state  = f['STATE_IN_ADDRESS'] || '';
  const zip    = f['ZIP_CODE_IN_ADDRESS'] || '';
  const addrParts = [street, city, state].filter(Boolean);
  const addr = (zip ? `${addrParts.join(', ')} ${zip}` : addrParts.join(', ')) || null;

  // Textract document type
  const idType = (f['ID_TYPE'] || '').toUpperCase();
  let documentType = null;
  if (idType.includes('PASSPORT'))            documentType = 'passport';
  else if (idType.includes('DRIVER'))         documentType = 'dl';
  else if (idType.includes('IDENTIFICATION')) documentType = 'state_id';

  return {
    data: {
      name:         [first, middle, last].filter(Boolean).join(' ') || f['NAME'] || null,
      dateOfBirth:  f['DATE_OF_BIRTH'] || null,
      address:      addr,
      idNumber:     f['DOCUMENT_NUMBER'] || f['ID_NUMBER'] || null,
      expiryDate:   f['DATE_OF_EXPIRY'] || f['EXPIRATION_DATE'] || null,
      issueDate:    f['DATE_OF_ISSUE']  || f['ISSUE_DATE'] || null,
      sex:          f['SEX'] || null,
      state:        f['STATE_NAME'] || state || null,
      documentType,
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

// ── Python microservice ────────────────────────────────────────────────────────
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

  const rawText   = ocrResult.raw_text || '';
  const confidence = ocrResult.confidence || 0;
  console.log(`[Python] confidence: ${typeof confidence === 'number' ? confidence.toFixed(1) : confidence}%`);
  console.log(`[Python] raw text:\n${rawText}`);

  const f = ocrResult.fields || {};
  const stateRaw = f.state || null;
  const state = (stateRaw && STATE_MAP[stateRaw.toUpperCase()])
    ? STATE_MAP[stateRaw.toUpperCase()]
    : stateRaw;

  const data = {
    name:         f.name         || null,
    dateOfBirth:  f.dateOfBirth  || null,
    address:      f.address      || null,
    idNumber:     f.idNumber     || null,
    expiryDate:   f.expiryDate   || null,
    issueDate:    f.issueDate    || null,
    sex:          f.sex          || null,
    state,
    documentType: ocrResult.documentType || null,
  };

  console.log('[Python] parsed fields:', JSON.stringify(data, null, 2));
  return { data, rawText };
}

// ── Merge two data objects — base wins, supplement fills null gaps ─────────────
function mergeData(base, supplement) {
  const merged = { ...base };
  for (const key of Object.keys(supplement)) {
    if (!merged[key] && supplement[key]) {
      merged[key] = supplement[key];
      console.log(`[Merge] '${key}' filled by Textract: ${supplement[key]}`);
    }
  }
  return merged;
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
    version: '8.0.0',
    ocr: 'docTR+PassportEye+PDF417 → Textract field fallback → error',
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
    console.log('\nReceived image:', (imageBuffer.length / 1024).toFixed(1), 'KB', side ? `| side=${side}` : '');

    // ── Step 1: Python (docTR + barcode + MRZ) — always primary ──────────────
    try {
      const result = await scanWithPython(imageBuffer, side);
      return res.json({ success: true, ...result, source: 'doctr+passporteye+barcode' });
    } catch (e) {
      console.warn('[server] Python service failed:', e.message);
    }

    // ── Step 2: Textract — only if Python service itself crashed ──────────────
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
OCR:      docTR+PassportEye+PDF417 → Textract field fallback → error
Python:   ${PYTHON_OCR_URL}
Textract: ${textractAvailable() ? 'available' : 'unavailable (set AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY)'}

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
