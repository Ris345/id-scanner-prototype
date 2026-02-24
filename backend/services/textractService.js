'use strict';

const { TextractClient, AnalyzeIDCommand } = require('@aws-sdk/client-textract');

let textractClient = null;

function isAvailable() {
  return !!(process.env.AWS_ACCESS_KEY_ID && process.env.AWS_SECRET_ACCESS_KEY);
}

function initTextract() {
  if (!isAvailable()) {
    console.warn('AWS credentials not set — Textract unavailable (will skip failsafe)');
    return;
  }
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
  if (!textractClient) throw new Error('Textract not initialized or AWS credentials not set');

  console.log('Sending to AWS Textract AnalyzeID...');
  const command = new AnalyzeIDCommand({
    DocumentPages: [{ Bytes: imageBuffer }],
  });

  const result = await textractClient.send(command);

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

  console.log('Textract fields:', fields);

  const data = {
    name: buildName(fields),
    dateOfBirth: fields['DATE_OF_BIRTH'] || null,
    address: fields['ADDRESS'] || buildAddress(fields),
    idNumber: fields['DOCUMENT_NUMBER'] || fields['ID_NUMBER'] || null,
    state: fields['STATE_NAME'] || fields['STATE'] || fields['PLACE_OF_BIRTH'] || null,
  };

  return { data, rawText: rawParts.join('\n') };
}

function buildAddress(fields) {
  const street = fields['ADDRESS'] || '';
  const city = fields['CITY_IN_ADDRESS'] || '';
  const state = fields['STATE_IN_ADDRESS'] || '';
  const zip = fields['ZIP_CODE_IN_ADDRESS'] || '';
  const county = fields['COUNTY'] || '';
  const parts = [street, city, county, state].filter(Boolean);
  const line = parts.join(', ');
  return (zip ? `${line} ${zip}` : line) || null;
}

function buildName(fields) {
  const first = fields['FIRST_NAME'] || '';
  const middle = fields['MIDDLE_NAME'] || '';
  const last = fields['LAST_NAME'] || fields['SURNAME'] || '';
  const full = [first, middle, last].filter(Boolean).join(' ');
  return full || fields['NAME'] || null;
}

module.exports = { initTextract, scanWithTextract, isAvailable };
