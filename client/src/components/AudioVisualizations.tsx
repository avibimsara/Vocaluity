"use client";

import type { Visualizations } from "@/lib/api";

const LABELS: Record<keyof Visualizations, string> = {
  waveform: "Waveform",
  mel_spectrogram: "Mel Spectrogram",
  mfccs: "MFCCs",
  chroma: "Chroma Features",
};

interface Props {
  data: Visualizations;
}

export default function AudioVisualizations({ data }: Props) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {(Object.keys(LABELS) as (keyof Visualizations)[]).map((key) => (
        <div key={key} className="bg-gray-800 rounded-lg p-3">
          <p className="text-sm text-gray-400 mb-2 font-medium">{LABELS[key]}</p>
          <img
            src={`data:image/png;base64,${data[key]}`}
            alt={LABELS[key]}
            className="w-full rounded"
          />
        </div>
      ))}
    </div>
  );
}
