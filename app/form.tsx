import { useState, useEffect } from 'react';
import { View, Text, StyleSheet, TextInput, TouchableOpacity, ScrollView, Alert, KeyboardAvoidingView, Platform } from 'react-native';
import { useRouter, Href } from 'expo-router';
import { useScan } from './context/ScanContext';

interface FormData {
  name: string;
  dateOfBirth: string;
  address: string;
  idNumber: string;
  state: string;
}

export default function Form() {
  const router = useRouter();
  const { scannedData } = useScan();
  const [submitting, setSubmitting] = useState(false);

  const [form, setForm] = useState<FormData>({
    name: '',
    dateOfBirth: '',
    address: '',
    idNumber: '',
    state: '',
  });

  // Populate form with scanned data
  useEffect(() => {
    if (scannedData) {
      setForm({
        name: scannedData.name || '',
        dateOfBirth: scannedData.dateOfBirth || '',
        address: scannedData.address || '',
        idNumber: scannedData.idNumber || '',
        state: scannedData.state || '',
      });
    }
  }, [scannedData]);

  const updateField = (field: keyof FormData, value: string) => {
    setForm(prev => ({ ...prev, [field]: value }));
  };

  const handleSubmit = async () => {
    setSubmitting(true);

    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000));

    console.log('Form submitted:', form);

    Alert.alert(
      'Success',
      'Form submitted successfully!',
      [
        {
          text: 'OK',
          onPress: () => router.replace('/' as Href),
        },
      ]
    );

    setSubmitting(false);
  };

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
    >
      <ScrollView style={styles.scroll} contentContainerStyle={styles.scrollContent}>
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity onPress={() => router.back()}>
            <Text style={styles.backText}>← Back</Text>
          </TouchableOpacity>
          <Text style={styles.title}>Verify Details</Text>
          <Text style={styles.subtitle}>Review and edit the extracted information</Text>
        </View>

        {/* Form Fields */}
        <View style={styles.formSection}>
          <FormField
            label="Full Name"
            value={form.name}
            onChangeText={(v) => updateField('name', v)}
            placeholder="John Smith"
          />

          <FormField
            label="Date of Birth"
            value={form.dateOfBirth}
            onChangeText={(v) => updateField('dateOfBirth', v)}
            placeholder="MM/DD/YYYY"
          />

          <FormField
            label="Address"
            value={form.address}
            onChangeText={(v) => updateField('address', v)}
            placeholder="123 Main St, City, State 12345"
            multiline
          />

          <FormField
            label="ID Number"
            value={form.idNumber}
            onChangeText={(v) => updateField('idNumber', v)}
            placeholder="D1234567"
          />

          <FormField
            label="State"
            value={form.state}
            onChangeText={(v) => updateField('state', v)}
            placeholder="California"
          />
        </View>

        {/* Raw OCR Text (for debugging) */}
        {scannedData?.rawText && (
          <View style={styles.rawSection}>
            <Text style={styles.rawLabel}>Raw OCR Text:</Text>
            <Text style={styles.rawText}>{scannedData.rawText}</Text>
          </View>
        )}

        {/* Submit Button */}
        <TouchableOpacity
          style={[styles.submitButton, submitting && styles.submitButtonDisabled]}
          onPress={handleSubmit}
          disabled={submitting}
        >
          <Text style={styles.submitButtonText}>
            {submitting ? 'Submitting...' : 'Submit'}
          </Text>
        </TouchableOpacity>

        {/* Rescan option */}
        <TouchableOpacity style={styles.rescanButton} onPress={() => router.replace('/scan' as Href)}>
          <Text style={styles.rescanText}>Scan Again</Text>
        </TouchableOpacity>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

function FormField({
  label,
  value,
  onChangeText,
  placeholder,
  multiline,
}: {
  label: string;
  value: string;
  onChangeText: (text: string) => void;
  placeholder?: string;
  multiline?: boolean;
}) {
  const hasValue = value.trim().length > 0;

  return (
    <View style={styles.fieldContainer}>
      <Text style={styles.label}>{label}</Text>
      <TextInput
        style={[
          styles.input,
          multiline && styles.inputMultiline,
          hasValue && styles.inputFilled,
        ]}
        value={value}
        onChangeText={onChangeText}
        placeholder={placeholder}
        placeholderTextColor="#555"
        multiline={multiline}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  scroll: {
    flex: 1,
  },
  scrollContent: {
    padding: 20,
    paddingTop: 60,
    paddingBottom: 40,
  },
  header: {
    marginBottom: 32,
  },
  backText: {
    color: '#007AFF',
    fontSize: 16,
    marginBottom: 16,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    color: '#888',
  },
  formSection: {
    gap: 16,
  },
  fieldContainer: {
    marginBottom: 4,
  },
  label: {
    fontSize: 14,
    color: '#888',
    marginBottom: 6,
  },
  input: {
    backgroundColor: '#111',
    borderRadius: 10,
    padding: 14,
    fontSize: 16,
    color: '#fff',
    borderWidth: 1,
    borderColor: '#222',
  },
  inputMultiline: {
    minHeight: 80,
    textAlignVertical: 'top',
  },
  inputFilled: {
    borderColor: '#007AFF',
  },
  rawSection: {
    marginTop: 24,
    padding: 12,
    backgroundColor: '#111',
    borderRadius: 8,
  },
  rawLabel: {
    fontSize: 12,
    color: '#666',
    marginBottom: 8,
  },
  rawText: {
    fontSize: 11,
    color: '#444',
    fontFamily: 'monospace',
  },
  submitButton: {
    backgroundColor: '#007AFF',
    paddingVertical: 18,
    borderRadius: 12,
    alignItems: 'center',
    marginTop: 32,
  },
  submitButtonDisabled: {
    backgroundColor: '#004999',
  },
  submitButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
  },
  rescanButton: {
    alignItems: 'center',
    marginTop: 16,
    paddingVertical: 12,
  },
  rescanText: {
    color: '#007AFF',
    fontSize: 16,
  },
});
