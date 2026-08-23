import { useState, useRef, useCallback } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ActivityIndicator } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import type { CameraView as CameraViewType } from 'expo-camera';
import { GestureDetector, Gesture } from 'react-native-gesture-handler';
import * as ImagePicker from 'expo-image-picker';
import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';
import { useRouter, Href } from 'expo-router';
import { scanID } from './utils/ocr';
import { useScan } from './context/ScanContext';

type ScanState = 'camera' | 'processing' | 'success' | 'error';
type Rect = { x: number; y: number; w: number; h: number };

// Keep a margin around the guide frame so a card the user framed slightly loose
// isn't clipped — the barcode sits right at the card edge. The server still runs
// its own card detection on whatever we send.
const CROP_PADDING = 0.12;

export default function Scan() {
  const router = useRouter();
  const { setScannedData } = useScan();
  const [permission, requestPermission] = useCameraPermissions();
  const [state, setState] = useState<ScanState>('camera');
  const [errorMsg, setErrorMsg] = useState('');
  const cameraRef = useRef<CameraViewType>(null);

  // Measured in window coords so the on-screen guide frame can be mapped onto the
  // captured photo. The frame used to be purely decorative — the full camera frame
  // was sent, leaving the card a small region inside a tall photo.
  const containerRef = useRef<View>(null);
  const frameRef = useRef<View>(null);
  const cameraRect = useRef<Rect | null>(null);
  const guideRect = useRef<Rect | null>(null);

  const measure = (ref: React.RefObject<View | null>, into: React.MutableRefObject<Rect | null>) => () =>
    ref.current?.measureInWindow((x, y, w, h) => {
      if (w > 0 && h > 0) into.current = { x, y, w, h };
    });

  // Zoom state: default 0.03 for a slight zoom to help with ID scanning
  const DEFAULT_ZOOM = 0.03;
  const [zoom, setZoom] = useState(DEFAULT_ZOOM);
  const zoomAtPinchStart = useRef(DEFAULT_ZOOM);

  const pinchGesture = Gesture.Pinch()
    .onStart(() => {
      zoomAtPinchStart.current = zoom;
    })
    .onUpdate((e) => {
      // Scale the zoom relative to the starting value
      const newZoom = zoomAtPinchStart.current * e.scale;
      setZoom(Math.min(Math.max(newZoom, 0), 1));
    });

  const processImage = async (uri: string) => {
    setState('processing');
    setErrorMsg('');

    try {
      const parsed = await scanID(uri);

      const hasAnyField = parsed.name || parsed.dateOfBirth || parsed.idNumber ||
                          parsed.expiryDate || parsed.address || parsed.sex;
      const hasRawText  = (parsed.rawText || '').trim().length > 20;
      console.log('[processImage] hasAnyField:', !!hasAnyField, '| hasRawText:', hasRawText);
      if (!hasAnyField && !hasRawText) {
        setState('error');
        setErrorMsg('No text detected. Please try again with a clearer image.');
        return;
      }

      setScannedData(parsed);
      setState('success');

      // Navigate to form after showing checkmark
      setTimeout(() => {
        router.replace('/form' as Href);
      }, 1200);
    } catch (err: any) {
      console.error('[processImage] Caught error:', err?.message ?? err);
      console.error('[processImage] Stack:', err?.stack);
      setState('error');
      setErrorMsg('Failed to process image. Please try again.');
    }
  };

  /**
   * Crop the capture down to the on-screen guide frame (plus padding).
   * Returns the original URI unchanged if the geometry can't be trusted — a bad crop
   * is far worse than none, since the server can still locate the card itself.
   */
  const cropToGuideFrame = async (photo: { uri: string; width: number; height: number }) => {
    const cam = cameraRect.current;
    const guide = guideRect.current;
    if (!cam || !guide || !photo.width || !photo.height) return photo.uri;

    // takePictureAsync may hand back a landscape buffer on some devices. Mapping a
    // portrait preview onto it would crop the wrong region, so bail instead.
    if (cam.h >= cam.w !== photo.height >= photo.width) {
      console.log('[crop] orientation mismatch — sending full frame');
      return photo.uri;
    }

    // The preview fills the view "cover"-style: scaled until both axes are covered, then centred.
    const scale = Math.max(cam.w / photo.width, cam.h / photo.height);
    if (!isFinite(scale) || scale <= 0) return photo.uri;
    const offsetX = (photo.width * scale - cam.w) / 2;
    const offsetY = (photo.height * scale - cam.h) / 2;

    const padX = guide.w * CROP_PADDING;
    const padY = guide.h * CROP_PADDING;

    // Guide frame -> view coords -> photo pixels, then clamp to the image.
    let originX = (guide.x - cam.x - padX + offsetX) / scale;
    let originY = (guide.y - cam.y - padY + offsetY) / scale;
    let width = (guide.w + padX * 2) / scale;
    let height = (guide.h + padY * 2) / scale;

    originX = Math.max(0, Math.min(originX, photo.width - 1));
    originY = Math.max(0, Math.min(originY, photo.height - 1));
    width = Math.max(1, Math.min(width, photo.width - originX));
    height = Math.max(1, Math.min(height, photo.height - originY));

    // Degenerate result means the maths went wrong somewhere — don't ship it.
    if (width < photo.width * 0.2 || height < photo.height * 0.05) {
      console.log('[crop] rect looks wrong — sending full frame');
      return photo.uri;
    }

    try {
      const rendered = await ImageManipulator.manipulate(photo.uri)
        .crop({ originX: Math.round(originX), originY: Math.round(originY),
                width: Math.round(width), height: Math.round(height) })
        .renderAsync();
      const result = await rendered.saveAsync({ format: SaveFormat.JPEG, compress: 0.95 });
      console.log(`[crop] ${photo.width}x${photo.height} -> ${Math.round(width)}x${Math.round(height)}`);
      return result.uri;
    } catch (err) {
      console.warn('[crop] failed, sending full frame:', err);
      return photo.uri;
    }
  };

  const takePhoto = async () => {
    if (cameraRef.current) {
      const photo = await cameraRef.current.takePictureAsync({ quality: 1 });
      if (photo) {
        await processImage(await cropToGuideFrame(photo));
      }
    }
  };

  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      quality: 1,
    });

    if (!result.canceled && result.assets[0]) {
      await processImage(result.assets[0].uri);
    }
  };

  const retry = () => {
    setState('camera');
    setErrorMsg('');
  };

  // Loading permission state
  if (!permission) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#007AFF" />
      </View>
    );
  }

  // Permission not granted - request it
  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.permissionText}>Camera access is needed to scan IDs</Text>
        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Grant Permission</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.buttonSecondary} onPress={() => router.back()}>
          <Text style={styles.buttonSecondaryText}>Go Back</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // Processing state
  if (state === 'processing') {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#007AFF" />
        <Text style={styles.processingText}>Processing ID...</Text>
      </View>
    );
  }

  // Success state with checkmark
  if (state === 'success') {
    return (
      <View style={styles.container}>
        <View style={styles.checkCircle}>
          <Text style={styles.checkmark}>✓</Text>
        </View>
        <Text style={styles.successText}>Scan Complete</Text>
      </View>
    );
  }

  // Error state
  if (state === 'error') {
    return (
      <View style={styles.container}>
        <View style={styles.errorCircle}>
          <Text style={styles.errorIcon}>✕</Text>
        </View>
        <Text style={styles.errorText}>{errorMsg}</Text>
        <TouchableOpacity style={styles.button} onPress={retry}>
          <Text style={styles.buttonText}>Try Again</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.buttonSecondary} onPress={() => router.back()}>
          <Text style={styles.buttonSecondaryText}>Cancel</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // Camera view
  return (
    <View style={styles.cameraContainer} ref={containerRef} onLayout={measure(containerRef, cameraRect)}>
      <GestureDetector gesture={pinchGesture}>
        <CameraView ref={cameraRef} style={styles.camera} facing="back" zoom={zoom}>
          <View style={styles.overlay}>
            {/* Top bar */}
            <View style={styles.topBar}>
              <TouchableOpacity onPress={() => router.back()}>
                <Text style={styles.cancelText}>Cancel</Text>
              </TouchableOpacity>
            </View>

            {/* Scan frame */}
            <View style={styles.frameContainer}>
              <View style={styles.scanFrame} ref={frameRef} onLayout={measure(frameRef, guideRect)}>
                <View style={[styles.corner, styles.topLeft]} />
                <View style={[styles.corner, styles.topRight]} />
                <View style={[styles.corner, styles.bottomLeft]} />
                <View style={[styles.corner, styles.bottomRight]} />
              </View>
              <Text style={styles.hint}>Position your ID within the frame{'\n'}Pinch to zoom</Text>
            </View>

            {/* Bottom controls */}
            <View style={styles.controls}>
              <TouchableOpacity style={styles.galleryButton} onPress={pickImage}>
                <Text style={styles.galleryText}>Gallery</Text>
              </TouchableOpacity>

              <TouchableOpacity style={styles.captureButton} onPress={takePhoto}>
                <View style={styles.captureInner} />
              </TouchableOpacity>

              <View style={styles.placeholder} />
            </View>
          </View>
        </CameraView>
      </GestureDetector>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#000',
    padding: 20,
  },
  cameraContainer: {
    flex: 1,
  },
  camera: {
    flex: 1,
  },
  overlay: {
    flex: 1,
    backgroundColor: 'transparent',
  },
  topBar: {
    paddingTop: 60,
    paddingHorizontal: 20,
    alignItems: 'flex-start',
  },
  cancelText: {
    color: '#fff',
    fontSize: 18,
  },
  frameContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  scanFrame: {
    width: 320,
    height: 200,
    position: 'relative',
  },
  corner: {
    position: 'absolute',
    width: 30,
    height: 30,
    borderColor: '#007AFF',
  },
  topLeft: {
    top: 0,
    left: 0,
    borderTopWidth: 3,
    borderLeftWidth: 3,
  },
  topRight: {
    top: 0,
    right: 0,
    borderTopWidth: 3,
    borderRightWidth: 3,
  },
  bottomLeft: {
    bottom: 0,
    left: 0,
    borderBottomWidth: 3,
    borderLeftWidth: 3,
  },
  bottomRight: {
    bottom: 0,
    right: 0,
    borderBottomWidth: 3,
    borderRightWidth: 3,
  },
  hint: {
    color: '#fff',
    fontSize: 16,
    marginTop: 24,
    textAlign: 'center',
  },
  controls: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 40,
    paddingBottom: 50,
  },
  galleryButton: {
    width: 70,
    alignItems: 'center',
  },
  galleryText: {
    color: '#fff',
    fontSize: 14,
  },
  captureButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'rgba(255,255,255,0.3)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  captureInner: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: '#fff',
  },
  placeholder: {
    width: 70,
  },
  permissionText: {
    color: '#888',
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 24,
  },
  button: {
    backgroundColor: '#007AFF',
    paddingVertical: 16,
    paddingHorizontal: 40,
    borderRadius: 12,
    marginBottom: 12,
  },
  buttonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
  },
  buttonSecondary: {
    paddingVertical: 16,
  },
  buttonSecondaryText: {
    color: '#007AFF',
    fontSize: 18,
  },
  processingText: {
    color: '#888',
    fontSize: 18,
    marginTop: 20,
  },
  checkCircle: {
    width: 120,
    height: 120,
    borderRadius: 60,
    backgroundColor: '#00C853',
    justifyContent: 'center',
    alignItems: 'center',
  },
  checkmark: {
    color: '#fff',
    fontSize: 60,
    fontWeight: 'bold',
  },
  successText: {
    color: '#fff',
    fontSize: 24,
    fontWeight: '600',
    marginTop: 24,
  },
  errorCircle: {
    width: 120,
    height: 120,
    borderRadius: 60,
    backgroundColor: '#FF3B30',
    justifyContent: 'center',
    alignItems: 'center',
  },
  errorIcon: {
    color: '#fff',
    fontSize: 60,
    fontWeight: 'bold',
  },
  errorText: {
    color: '#888',
    fontSize: 16,
    textAlign: 'center',
    marginTop: 24,
    marginBottom: 24,
    paddingHorizontal: 20,
  },
});
