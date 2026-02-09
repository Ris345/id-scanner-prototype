// Universal ID Parser - extracts structured data from OCR text

export interface ParsedID {
  name: string | null;
  dateOfBirth: string | null;
  address: string | null;
  idNumber: string | null;
  state: string | null;
  rawText: string;
}

export function parseIDText(ocrText: string): ParsedID {
  console.log('=== RAW OCR TEXT ===');
  console.log(ocrText);
  console.log('====================');

  const lines = ocrText.split('\n').map(l => l.trim()).filter(Boolean);
  const upperText = ocrText.toUpperCase();

  // Check for MRZ first (passports)
  const mrzData = parseMRZ(lines);
  if (mrzData) {
    console.log('MRZ detected:', mrzData);
    return { ...mrzData, rawText: ocrText };
  }

  // Extract all fields
  const allDates = extractAllDates(ocrText);
  console.log('Found dates:', allDates);

  const name = extractName(lines, upperText);
  console.log('Found name:', name);

  const address = extractAddress(lines, upperText);
  console.log('Found address:', address);

  const idNumber = extractIDNumber(lines, upperText);
  console.log('Found ID number:', idNumber);

  const state = extractState(upperText);
  console.log('Found state:', state);

  const result: ParsedID = {
    name: name,
    dateOfBirth: allDates[0] || null,
    address: address,
    idNumber: idNumber,
    state: state,
    rawText: ocrText,
  };

  console.log('=== PARSED RESULT ===');
  console.log(result);
  console.log('=====================');

  return result;
}

function parseMRZ(lines: string[]): Omit<ParsedID, 'rawText'> | null {
  const MRZ_PATTERN = /^[A-Z0-9<]{30,44}$/;
  const mrzLines = lines.filter(line => MRZ_PATTERN.test(line.replace(/\s/g, '')));

  if (mrzLines.length < 2) return null;

  const line1 = mrzLines[0].replace(/\s/g, '');
  const line2 = mrzLines[1].replace(/\s/g, '');

  if (line1.length >= 30) {
    const namePart = line1.substring(5).split('<<');
    const lastName = namePart[0]?.replace(/</g, ' ').trim() || '';
    const firstName = namePart[1]?.replace(/</g, ' ').trim() || '';
    const fullName = [firstName, lastName].filter(Boolean).join(' ') || null;

    let dob = null;
    if (line2.length >= 20) {
      const dobRaw = line2.substring(0, 6);
      if (/^\d{6}$/.test(dobRaw)) {
        const year = parseInt(dobRaw.substring(0, 2));
        const fullYear = year > 30 ? 1900 + year : 2000 + year;
        dob = `${fullYear}-${dobRaw.substring(2, 4)}-${dobRaw.substring(4, 6)}`;
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

  return null;
}

function extractAllDates(text: string): string[] {
  const dates: string[] = [];

  // Pattern: MM/DD/YYYY or DD/MM/YYYY
  const datePattern1 = /\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})\b/g;
  let match;
  while ((match = datePattern1.exec(text)) !== null) {
    dates.push(match[0]);
  }

  // Pattern: YYYY-MM-DD
  const datePattern2 = /\b(\d{4})[\/\-](\d{1,2})[\/\-](\d{1,2})\b/g;
  while ((match = datePattern2.exec(text)) !== null) {
    dates.push(match[0]);
  }

  // Pattern: 01 JAN 1990
  const datePattern3 = /\b(\d{1,2})[\s\-]?(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*[\s\-,]*(\d{2,4})\b/gi;
  while ((match = datePattern3.exec(text)) !== null) {
    dates.push(match[0]);
  }

  return dates;
}

function extractName(lines: string[], upperText: string): string | null {
  // Strategy 1: Look for labeled name fields
  for (const line of lines) {
    const upper = line.toUpperCase();

    if (upper.match(/^NAME\b/) || upper.includes('FULL NAME')) {
      const value = extractValueAfterLabel(line);
      if (value && isNameLike(value)) {
        return value;
      }
    }
  }

  // Strategy 2: Look for FIRST + LAST name labels
  let firstName: string | null = null;
  let lastName: string | null = null;

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

  if (firstName || lastName) {
    return [firstName, lastName].filter(Boolean).join(' ');
  }

  // Strategy 3: Find lines that look like names (2-3 words, no numbers)
  for (const line of lines) {
    if (/\d/.test(line)) continue;
    if (line.length < 4) continue;
    if (/^(NAME|DOB|SEX|ADDRESS|LICENSE|EXP|CLASS|ISS|HT|WT|EYES|HAIR|STATE|DL)$/i.test(line.trim())) continue;

    const words = line.split(/\s+/).filter(w => /^[A-Za-z]+$/.test(w) && w.length > 1);
    if (words.length >= 2 && words.length <= 4) {
      return words.join(' ');
    }
  }

  return null;
}

function extractValueAfterLabel(line: string): string | null {
  const parts = line.split(/[:\s]+/);
  if (parts.length >= 2) {
    return parts.slice(1).join(' ').trim() || null;
  }
  return null;
}

function isNameLike(text: string): boolean {
  return /^[A-Za-z\s\-']+$/.test(text) && text.length >= 2 && text.length <= 50;
}

function extractAddress(lines: string[], upperText: string): string | null {
  const addressLines: string[] = [];
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
      if (line.length > 5) {
        addressLines.push(line);
        capturing = true;
      }
      if (ZIP_CODE.test(line)) break;
    }
  }

  return addressLines.length > 0 ? addressLines.join(', ') : null;
}

function extractIDNumber(lines: string[], upperText: string): string | null {
  // Look for labeled ID numbers
  for (const line of lines) {
    const upper = line.toUpperCase();
    if (upper.includes('DL') || upper.includes('LICENSE') || upper.includes('ID NO') ||
        upper.includes('DOCUMENT') || upper.includes('NUMBER') || upper.includes('NO.') ||
        upper.includes('ID#') || upper.includes('DLN')) {
      const match = line.match(/[A-Z]?\d{5,}/i) || line.match(/[A-Z0-9]{7,15}/i);
      if (match) return match[0];
    }
  }

  // Look for standalone patterns
  for (const line of lines) {
    const match = line.match(/\b[A-Z]\d{7,}\b/i);
    if (match) return match[0];
  }

  for (const line of lines) {
    const match = line.match(/\b[A-Z0-9]{8,12}\b/);
    if (match && !/^\d+$/.test(match[0])) {
      return match[0];
    }
  }

  return null;
}

function extractState(text: string): string | null {
  const STATE_MAP: { [key: string]: string } = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
    'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
    'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
    'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
    'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
    'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
    'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
    'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
  };

  // Look for full state names first
  for (const [abbr, name] of Object.entries(STATE_MAP)) {
    if (text.includes(name.toUpperCase())) {
      return name;
    }
  }

  // Look for state abbreviations
  const abbrevMatch = text.match(/\b(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)\b/);
  if (abbrevMatch) {
    return STATE_MAP[abbrevMatch[1]] || abbrevMatch[1];
  }

  return null;
}
