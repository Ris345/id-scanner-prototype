import { Stack } from 'expo-router';
import { ScanProvider } from './context/ScanContext';

export default function RootLayout() {
  return (
    <ScanProvider>
      <Stack screenOptions={{ headerShown: false }}>
        <Stack.Screen name="index" />
        <Stack.Screen name="scan" />
        <Stack.Screen name="form" />
      </Stack>
    </ScanProvider>
  );
}
