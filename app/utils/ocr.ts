import { Platform } from 'react-native';
import { ParsedID } from './idParser';

// Production: deployed backend on Render
// Development: set your own backend URL below (run your own backend with your own AWS credentials)
const API_URL = __DEV__
  ? 'http://localhost:3001'
  : 'https://id-scanner-prototype.onrender.com';

/**
 * Send image to backend for ID extraction.
 * @param imageUri  Local image URI
 * @param side      Optional hint: 'front' | 'back' — skips irrelevant processing on each side
 */
export async function scanID(imageUri: string, side?: 'front' | 'back'): Promise<ParsedID> {
  const base64Image = await imageToBase64(imageUri);

  const body: Record<string, string> = { image: base64Image };
  if (side) body.side = side;

  const response = await fetch(`${API_URL}/api/scan`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.message || 'Failed to scan ID');
  }

  const result = await response.json();

  return {
    name:         result.data.name         ?? null,
    dateOfBirth:  result.data.dateOfBirth  ?? null,
    address:      result.data.address      ?? null,
    idNumber:     result.data.idNumber     ?? null,
    state:        result.data.state        ?? null,
    expiryDate:   result.data.expiryDate   ?? null,
    issueDate:    result.data.issueDate    ?? null,
    sex:          result.data.sex          ?? null,
    documentType: result.data.documentType ?? null,
    rawText:      result.rawText           ?? '',
  };
}

async function imageToBase64(uri: string): Promise<string> {
  if (Platform.OS === 'web') {
    // For web: fetch the blob and convert
    const response = await fetch(uri);
    const blob = await response.blob();
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  } else {
    // For native: use expo-file-system
    const FileSystem = require('expo-file-system');
    const base64 = await FileSystem.readAsStringAsync(uri, {
      encoding: FileSystem.EncodingType.Base64,
    });
    return `data:image/jpeg;base64,${base64}`;
  }
}
