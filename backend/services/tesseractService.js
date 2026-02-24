'use strict';

const path = require('path');
const { createWorker } = require('tesseract.js');

let worker = null;

async function initWorker() {
  worker = await createWorker('eng', 1, {
    langPath: path.join(__dirname, '..'),
  });
  console.log('Tesseract worker initialized');
}

// mrz v5 is ESM-only — use dynamic import from CJS
let _mrzParse = null;
async function getMrzParse() {
  if (_mrzParse) return _mrzParse;
  try {
    const m = await import('mrz');
    _mrzParse = typeof m.parse === 'function' ? m.parse : (m.default && m.default.parse) || null;
  } catch (e) {
    console.warn('mrz package unavailable:', e.message);
    _mrzParse = false; // mark as unavailable so we don't retry
  }
  return _mrzParse || null;
}

async function scanWithTesseract(imageBuffer) {
  if (!worker) throw new Error('Tesseract worker not initialized');

  const result = await worker.recognize(imageBuffer);
  const { text, confidence } = result.data;

  const rawText = text;
  const lines = text.split('\n').map(l => l.trim()).filter(Boolean);

  let data = null;

  // Try MRZ lines (passports / ID cards with machine-readable zone)
  const MRZ_PATTERN = /^[A-Z0-9<]{20,44}$/;
  const mrzLines = lines.filter(l => MRZ_PATTERN.test(l.replace(/\s/g, '')));

  if (mrzLines.length >= 2) {
    const parse = await getMrzParse();
    if (parse) {
      try {
        const mrzResult = parse(mrzLines.slice(0, 3));
        const fields = mrzResult.fields || {};

        if (mrzResult.valid || Object.values(fields).some(Boolean)) {
          data = {
            name: buildMrzName(fields),
            dateOfBirth: convertMrzDate(fields.birthDate),
            address: extractAddress(lines, text.toUpperCase()), // MRZ has no address; regex fallback
            idNumber: fields.documentNumber ? fields.documentNumber.replace(/</g, '') : null,
            state: fields.issuingState || null,
          };
        }
      } catch (e) {
        console.warn('mrz.parse failed:', e.message);
      }
    }

    // If mrz lib failed/unavailable, parse MRZ manually
    if (!data) {
      data = parseManualMRZ(mrzLines);
    }
  }

  // No MRZ found — fall back to regex parsers
  if (!data) {
    const upperText = text.toUpperCase();
    const allDates = extractAllDates(text);
    data = {
      name: extractName(lines, upperText),
      dateOfBirth: allDates[0] || null,
      address: extractAddress(lines, upperText),
      idNumber: extractIDNumber(lines, upperText),
      state: extractState(upperText),
    };
  }

  // Throw to trigger Textract failsafe if result is too poor
  const allNull = Object.values(data).every(v => v === null);
  if (confidence < 40 || allNull) {
    throw new Error(
      `Tesseract quality too low (confidence: ${confidence.toFixed(1)}, all fields null: ${allNull})`
    );
  }

  return { data, rawText };
}

// ── MRZ helpers ─────────────────────────────────────────────────────────────

function buildMrzName(fields) {
  const last = (fields.lastName || '').replace(/</g, ' ').trim();
  const first = (fields.firstName || '').replace(/</g, ' ').trim();
  return [first, last].filter(Boolean).join(' ') || null;
}

function convertMrzDate(yymmdd) {
  if (!yymmdd || !/^\d{6}$/.test(yymmdd)) return null;
  const year = parseInt(yymmdd.substring(0, 2), 10);
  const fullYear = year > 30 ? 1900 + year : 2000 + year;
  const month = yymmdd.substring(2, 4);
  const day = yymmdd.substring(4, 6);
  return `${month}/${day}/${fullYear}`;
}

function parseManualMRZ(mrzLines) {
  const line1 = mrzLines[0].replace(/\s/g, '');
  const line2 = mrzLines[1].replace(/\s/g, '');

  const namePart = line1.substring(5).split('<<');
  const lastName = (namePart[0] || '').replace(/</g, ' ').trim();
  const firstName = (namePart[1] || '').replace(/</g, ' ').trim();
  const fullName = [firstName, lastName].filter(Boolean).join(' ') || null;

  let dob = null;
  if (line2.length >= 6) {
    const dobRaw = line2.substring(0, 6);
    if (/^\d{6}$/.test(dobRaw)) {
      const year = parseInt(dobRaw.substring(0, 2), 10);
      const fullYear = year > 30 ? 1900 + year : 2000 + year;
      const month = dobRaw.substring(2, 4);
      const day = dobRaw.substring(4, 6);
      dob = `${month}/${day}/${fullYear}`;
    }
  }

  return {
    name: fullName,
    dateOfBirth: dob,
    address: null,
    idNumber: line2.substring(0, 9).replace(/</g, '') || null,
    state: null,
  };
}

// ── Regex parsers (ported from app/utils/idParser.ts) ───────────────────────

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

module.exports = { initWorker, scanWithTesseract };
