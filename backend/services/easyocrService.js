'use strict';

const PYTHON_OCR_URL = process.env.PYTHON_OCR_URL || 'http://localhost:3002';

async function isHealthy() {
  try {
    const res = await fetch(`${PYTHON_OCR_URL}/health`, {
      signal: AbortSignal.timeout(3000),
    });
    return res.ok;
  } catch {
    return false;
  }
}

async function scanWithEasyOCR(imageBuffer) {
  const base64 = imageBuffer.toString('base64');

  const res = await fetch(`${PYTHON_OCR_URL}/ocr`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: base64 }),
    signal: AbortSignal.timeout(30000),
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Python OCR service error ${res.status}: ${body}`);
  }

  const ocrResult = await res.json();
  if (!ocrResult.success) {
    throw new Error(`Python OCR failed: ${ocrResult.error || 'unknown error'}`);
  }

  const rawText = ocrResult.raw_text;
  const confidence = ocrResult.confidence;
  const lines = (ocrResult.lines || []).map(l => l.text);

  console.log(`[easyocrService] Python OCR confidence: ${confidence.toFixed(1)}%`);
  console.log(`[easyocrService] Raw text received:\n${rawText}`);

  let data = null;

  // ── Path 1: PassportEye found an MRZ zone ────────────────────────────────
  if (ocrResult.mrz) {
    console.log('[easyocrService] MRZ detected — using PassportEye structured fields');
    console.log('[easyocrService] MRZ raw fields:', JSON.stringify(ocrResult.mrz, null, 2));

    const mrz = ocrResult.mrz;
    const surname = (mrz.surname || '').replace(/</g, ' ').trim();
    const givenNames = (mrz.given_names || '').replace(/</g, ' ').trim();
    const name = [givenNames, surname].filter(Boolean).join(' ') || null;

    data = {
      name,
      dateOfBirth: convertMrzDate(mrz.date_of_birth),
      address: null,
      idNumber: (mrz.document_number || '').replace(/</g, '').trim() || null,
      state: mrz.country || mrz.nationality || null,
    };

    console.log('[easyocrService] Parsed from MRZ:', JSON.stringify(data, null, 2));
  }

  // ── Path 2: No MRZ — regex-parse EasyOCR raw text ────────────────────────
  if (!data) {
    console.log('[easyocrService] No MRZ — falling back to regex parsing on EasyOCR text');
    const upperText = rawText.toUpperCase();
    const allDates = extractAllDates(rawText);
    data = {
      name: extractName(lines, upperText),
      dateOfBirth: allDates[0] || null,
      address: extractAddress(lines, upperText),
      idNumber: extractIDNumber(lines, upperText),
      state: extractState(upperText),
    };

    console.log('[easyocrService] Parsed from regex:', JSON.stringify(data, null, 2));
  }

  const allNull = Object.values(data).every(v => v === null);
  if (allNull) {
    console.warn('[easyocrService] All fields null — throwing to trigger Textract fallback');
    throw new Error(
      `EasyOCR quality too low (confidence: ${confidence.toFixed(1)}, all fields null)`
    );
  }

  return { data, rawText };
}

// ── MRZ date conversion (YYMMDD → MM/DD/YYYY) ────────────────────────────────

function convertMrzDate(yymmdd) {
  if (!yymmdd || !/^\d{6}$/.test(String(yymmdd))) return null;
  const s = String(yymmdd);
  const year = parseInt(s.substring(0, 2), 10);
  const fullYear = year > 30 ? 1900 + year : 2000 + year;
  const month = s.substring(2, 4);
  const day = s.substring(4, 6);
  return `${month}/${day}/${fullYear}`;
}

// ── Regex parsers (used when no MRZ is present) ───────────────────────────────

function extractAllDates(text) {
  const dates = [];
  let match;
  const p1 = /\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})\b/g;
  while ((match = p1.exec(text)) !== null) dates.push(match[0]);
  const p2 = /\b(\d{4})[\/\-](\d{1,2})[\/\-](\d{1,2})\b/g;
  while ((match = p2.exec(text)) !== null) dates.push(match[0]);
  const p3 = /\b(\d{1,2})[\s\-]?(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*[\s\-,]*(\d{2,4})\b/gi;
  while ((match = p3.exec(text)) !== null) dates.push(match[0]);
  return dates;
}

function extractName(lines, upperText) {
  for (const line of lines) {
    const upper = line.toUpperCase();
    if (upper.match(/^NAME\b/) || upper.includes('FULL NAME')) {
      const value = extractValueAfterLabel(line);
      if (value && isNameLike(value)) return value;
    }
  }

  let firstName = null, lastName = null;
  for (const line of lines) {
    const upper = line.toUpperCase();
    if (upper.includes('FIRST') || upper.includes('GIVEN') || upper.match(/\bFN\b/)) {
      const value = extractValueAfterLabel(line);
      if (value && isNameLike(value)) firstName = value;
    }
    if (upper.includes('LAST') || upper.includes('SURNAME') || upper.match(/\bLN\b/)) {
      const value = extractValueAfterLabel(line);
      if (value && isNameLike(value)) lastName = value;
    }
  }
  if (firstName || lastName) return [firstName, lastName].filter(Boolean).join(' ');

  for (const line of lines) {
    if (/\d/.test(line) || line.length < 4) continue;
    if (/^(NAME|DOB|SEX|ADDRESS|LICENSE|EXP|CLASS|ISS|HT|WT|EYES|HAIR|STATE|DL)$/i.test(line.trim())) continue;
    const words = line.split(/\s+/).filter(w => /^[A-Za-z]+$/.test(w) && w.length > 1);
    if (words.length >= 2 && words.length <= 4) return words.join(' ');
  }

  return null;
}

function extractValueAfterLabel(line) {
  const parts = line.split(/[:\s]+/);
  return parts.length >= 2 ? parts.slice(1).join(' ').trim() || null : null;
}

function isNameLike(text) {
  return /^[A-Za-z\s\-']+$/.test(text) && text.length >= 2 && text.length <= 50;
}

function extractAddress(lines, upperText) {
  const addressLines = [];
  let capturing = false;
  const US_STATES = /\b(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)\b/i;
  const ZIP_CODE = /\b\d{5}(-\d{4})?\b/;
  const STREET_PATTERN = /\d+\s+[\w\s]+(ST|STREET|AVE|AVENUE|RD|ROAD|DR|DRIVE|LN|LANE|BLVD|CT|COURT|WAY|PL|PLACE)/i;

  for (const line of lines) {
    const upper = line.toUpperCase();
    if (upper.includes('ADDRESS') || upper.includes('ADDR')) {
      capturing = true;
      const afterLabel = line.replace(/.*ADDRESS[:\s]*/i, '').trim();
      if (afterLabel) addressLines.push(afterLabel);
      continue;
    }
    if (capturing || STREET_PATTERN.test(line) || (US_STATES.test(line) && ZIP_CODE.test(line))) {
      if (line.length > 5) { addressLines.push(line); capturing = true; }
      if (ZIP_CODE.test(line)) break;
    }
  }

  return addressLines.length > 0 ? addressLines.join(', ') : null;
}

function extractIDNumber(lines, upperText) {
  for (const line of lines) {
    const upper = line.toUpperCase();
    if (upper.includes('DL') || upper.includes('LICENSE') || upper.includes('ID NO') ||
        upper.includes('DOCUMENT') || upper.includes('NUMBER') || upper.includes('NO.') ||
        upper.includes('ID#') || upper.includes('DLN')) {
      const match = line.match(/[A-Z]?\d{5,}/i) || line.match(/[A-Z0-9]{7,15}/i);
      if (match) return match[0];
    }
  }
  for (const line of lines) {
    const match = line.match(/\b[A-Z]\d{7,}\b/i);
    if (match) return match[0];
  }
  for (const line of lines) {
    const match = line.match(/\b[A-Z0-9]{8,12}\b/);
    if (match && !/^\d+$/.test(match[0])) return match[0];
  }
  return null;
}

function extractState(text) {
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
  for (const [, name] of Object.entries(STATE_MAP)) {
    if (text.includes(name.toUpperCase())) return name;
  }
  const abbrevMatch = text.match(/\b(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)\b/);
  if (abbrevMatch) return STATE_MAP[abbrevMatch[1]] || abbrevMatch[1];
  return null;
}

module.exports = { scanWithEasyOCR, isHealthy };
