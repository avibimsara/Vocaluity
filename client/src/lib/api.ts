const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface PredictionResult {
  prediction: string;
  predicted_class: number;
  confidence: number;
  probabilities: Record<string, number>;
}

export interface Visualizations {
  waveform: string;
  mel_spectrogram: string;
  mfccs: string;
  chroma: string;
}

export interface ModelInfo {
  model_name: string;
  accuracy: number;
  classes: string[];
}

export interface HealthStatus {
  status: string;
  device: string;
  model_loaded: boolean;
}

export async function predictAudio(file: File): Promise<PredictionResult> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/predict`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getVisualizations(file: File): Promise<Visualizations> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/visualize`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getModelInfo(): Promise<ModelInfo> {
  const res = await fetch(`${API_BASE}/model/info`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getHealth(): Promise<HealthStatus> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
