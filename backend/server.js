const express = require('express');
const multer = require('multer');
const cors = require('cors');
const { TextractClient, AnalyzeIDCommand } = require('@aws-sdk/client-textract');
const sharp = require('sharp');
const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '../.env') });

const app = express();
const PORT = 3001;

if (!process.env.AWS_ACCESS_KEY_ID || !process.env.AWS_SECRET_ACCESS_KEY) {
  console.error('AWS credentials not set in backend/.env');
  process.exit(1);
}

const textract = new TextractClient({
  region: process.env.AWS_REGION || 'us-east-1',
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
  },
});

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));

// In-memory storage for multer
const storage = multer.memoryStorage();
const upload = multer({ storage, limits: { fileSize: 50 * 1024 * 1024 } });

// Health check
app.get('/', (req, res) => {
  res.json({ status: 'ID Scanner API running', version: '3.0.0' });
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
    const processed = await sharp(imageBuffer)
      .greyscale()
      .normalize()
      .sharpen({ sigma: 2 })
      .resize({ width: 2000, withoutEnlargement: true })
      .png()
      .toBuffer();

    console.log('Preprocessed:', (processed.length / 1024).toFixed(1), 'KB');
    console.log('Sending to AWS Textract AnalyzeID...');

    const command = new AnalyzeIDCommand({
      DocumentPages: [{ Bytes: processed }],
    });

    const result = await textract.send(command);

    // Extract fields from Textract AnalyzeID response
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

    // Map Textract field names to our schema
    const parsedData = {
      name: buildName(fields),
      dateOfBirth: fields['DATE_OF_BIRTH'] || fields['EXPIRATION_DATE'] || null,
      address: fields['ADDRESS'] || buildAddress(fields),
      idNumber: fields['DOCUMENT_NUMBER'] || fields['ID_NUMBER'] || null,
      state: fields['STATE_NAME'] || fields['STATE'] || fields['PLACE_OF_BIRTH'] || null,
    };

    console.log('Parsed:', parsedData);

    res.json({
      success: true,
      data: parsedData,
      rawText: rawParts.join('\n'),
    });

  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({
      error: 'Failed to process image',
      message: error.message
    });
  }
});

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

app.listen(PORT, () => {
  console.log(`
ID Scanner Backend
━━━━━━━━━━━━━━━━━━━━━━
Server:  http://localhost:${PORT}
OCR:     AWS Textract AnalyzeID
Region:  ${process.env.AWS_REGION || 'us-east-1'}

Endpoints:
  GET  /           Health check
  POST /api/scan   Scan ID image
`);
});
