import { useState, useRef, useCallback } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ActivityIndicator } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import type { CameraView as CameraViewType } from 'expo-camera';
import { GestureDetector, Gesture } from 'react-native-gesture-handler';
import * as ImagePicker from 'expo-image-picker';
import { useRouter, Href } from 'expo-router';
import { scanID } from './utils/ocr';
import { useScan } from './context/ScanContext';

type ScanState = 'camera' | 'processing' | 'success' | 'error';

export default function Scan() {
  const router = useRouter();
  const { setScannedData } = useScan();
  const [permission, requestPermission] = useCameraPermissions();
  const [state, setState] = useState<ScanState>('camera');
  const [errorMsg, setErrorMsg] = useState('');
  const cameraRef = useRef<CameraViewType>(null);

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

      if (!parsed.name && !parsed.dateOfBirth && !parsed.idNumber) {
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
    } catch (err) {
      console.error(err);
      setState('error');
      setErrorMsg('Failed to process image. Please try again.');
    }
  };

  const takePhoto = async () => {
    if (cameraRef.current) {
      const photo = await cameraRef.current.takePictureAsync({ quality: 1 });
      if (photo) {
        await processImage(photo.uri);
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
    <View style={styles.cameraContainer}>
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
              <View style={styles.scanFrame}>
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
