import { Platform } from 'react-native';
import Constants from 'expo-constants';
import { ParsedID } from './idParser';

/**
 * In dev, resolve the backend from Metro's own host.
 *
 * On a physical phone `localhost` is the phone itself, so a hardcoded localhost URL can
 * never reach the dev machine. `hostUri` is where Metro is being served from — the Mac's
 * LAN IP — so reuse that host and swap Metro's port for the API's.
 * Simulators and web still resolve localhost fine, hence the fallback.
 *
 * This only works when Metro is served over the LAN. Under `expo start --tunnel`, hostUri
 * is an `.exp.direct` host that forwards 8081 and nothing else, so port 3001 there is dead.
 * Tunnelled runs must set EXPO_PUBLIC_API_URL to the backend's own public URL.
 */
function devApiUrl(): string {
  const override = process.env.EXPO_PUBLIC_API_URL;
  if (override) return override.replace(/\/+$/, '');

  const host = Constants.expoConfig?.hostUri?.split(':')[0];
  if (host && /\.exp\.direct$/.test(host)) {
    console.warn('[scanID] Metro is tunnelled but EXPO_PUBLIC_API_URL is unset — ' +
                 'the API is not reachable through the Expo tunnel.');
  }
  return host ? `http://${host}:3001` : 'http://localhost:3001';
}

// Production: deployed backend on Render
const API_URL = __DEV__ ? devApiUrl() : 'https://id-scanner-prototype.onrender.com';

/**
 * Send image to backend for ID extraction.
 * @param imageUri  Local image URI
 * @param side      Optional hint: 'front' | 'back' — skips irrelevant processing on each side
 */
export async function scanID(imageUri: string, side?: 'front' | 'back'): Promise<ParsedID> {
  console.log('[scanID] Starting — URI:', imageUri.slice(0, 60), '| side:', side ?? 'none');

  const base64Image = await imageToBase64(imageUri);
  console.log('[scanID] Base64 ready — length:', base64Image.length, '| posting to:', `${API_URL}/api/scan`);

  const platform = Platform.OS === 'web' ? 'desktop' : 'mobile';
  const body: Record<string, string> = { image: base64Image, platform };
  if (side) body.side = side;

  let response: Response;
  try {
    response = await fetch(`${API_URL}/api/scan`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        // Free ngrok serves an HTML interstitial to anything it reads as a browser,
        // which would arrive here as a JSON parse failure.
        'ngrok-skip-browser-warning': 'true',
      },
      body: JSON.stringify(body),
    });
  } catch (networkErr) {
    console.error('[scanID] Network error (fetch threw):', networkErr);
    throw networkErr;
  }

  console.log('[scanID] Response status:', response.status, response.statusText);

  if (!response.ok) {
    const errorBody = await response.text();
    console.error('[scanID] Non-2xx response body:', errorBody);
    let parsed: any = {};
    try { parsed = JSON.parse(errorBody); } catch {}
    throw new Error(parsed.message || parsed.error || `HTTP ${response.status}`);
  }

  const result = await response.json();
  console.log('[scanID] --- SUMMARY ---');
  console.log('[scanID] success:', result.success);
  console.log('[scanID] has data key:', 'data' in result);
  console.log('[scanID] documentType:', result.documentType);
  console.log('[scanID] rawText length:', (result.rawText ?? '').length);
  console.log('[scanID] rawText preview:', (result.rawText ?? '').slice(0, 300));
  console.log('[scanID] fields:', JSON.stringify(result.data));
  if (!result.success) {
    throw new Error(result.error || 'OCR service returned success=false');
  }

  return {
    name:         result.data?.name         ?? null,
    dateOfBirth:  result.data?.dateOfBirth  ?? null,
    address:      result.data?.address      ?? null,
    idNumber:     result.data?.idNumber     ?? null,
    state:        result.data?.state        ?? null,
    expiryDate:   result.data?.expiryDate   ?? null,
    issueDate:    result.data?.issueDate    ?? null,
    sex:          result.data?.sex          ?? null,
    documentType: result.data?.documentType ?? null,
    rawText:      result.rawText            ?? '',
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
