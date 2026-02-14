import { Stack } from 'expo-router';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { ScanProvider } from './context/ScanContext';

export default function RootLayout() {
  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <ScanProvider>
        <Stack screenOptions={{ headerShown: false }}>
          <Stack.Screen name="index" />
          <Stack.Screen name="scan" />
          <Stack.Screen name="form" />
        </Stack>
      </ScanProvider>
    </GestureHandlerRootView>
  );
}
