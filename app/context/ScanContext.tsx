import { createContext, useContext, useState, useCallback, ReactNode } from 'react';
import { ParsedID } from '../utils/idParser';

interface ScanContextType {
  scannedData: ParsedID | null;
  setScannedData: (data: ParsedID | null) => void;
  clearData: () => void;
}

const ScanContext = createContext<ScanContextType | null>(null);

export function ScanProvider({ children }: { children: ReactNode }) {
  const [scannedData, setScannedData] = useState<ParsedID | null>(null);

  const clearData = useCallback(() => setScannedData(null), []);

  return (
    <ScanContext.Provider value={{ scannedData, setScannedData, clearData }}>
      {children}
    </ScanContext.Provider>
  );
}

export function useScan() {
  const context = useContext(ScanContext);
  if (!context) {
    throw new Error('useScan must be used within ScanProvider');
  }
  return context;
}
