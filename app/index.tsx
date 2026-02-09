import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { useRouter, Href } from 'expo-router';
import { useScan } from './context/ScanContext';
import { useEffect } from 'react';

export default function Home() {
  const router = useRouter();
  const { clearData } = useScan();

  useEffect(() => {
    clearData();
  }, [clearData]);

  return (
    <View style={styles.container}>
      <Text style={styles.title}>ID Scanner</Text>
      <Text style={styles.subtitle}>Scan your government ID to auto-fill forms</Text>

      <TouchableOpacity style={styles.scanButton} onPress={() => router.push('/scan' as Href)}>
        <Text style={styles.scanButtonText}>Scan</Text>
      </TouchableOpacity>

      <Text style={styles.hint}>Supports passports, driver's licenses, and national IDs</Text>
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
  title: {
    fontSize: 42,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    color: '#888',
    textAlign: 'center',
    marginBottom: 60,
  },
  scanButton: {
    width: 180,
    height: 180,
    borderRadius: 90,
    backgroundColor: '#007AFF',
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#007AFF',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.5,
    shadowRadius: 20,
    elevation: 10,
  },
  scanButtonText: {
    color: '#fff',
    fontSize: 28,
    fontWeight: 'bold',
  },
  hint: {
    color: '#555',
    fontSize: 14,
    marginTop: 60,
    textAlign: 'center',
  },
});
