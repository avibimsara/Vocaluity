"use client";

import type { PredictionResult } from "@/lib/api";

interface Props {
  result: PredictionResult;
}

export default function ResultDisplay({ result }: Props) {
  const isReal = result.predicted_class === 0;

  return (
    <div
      className={`rounded-xl p-6 text-center text-xl font-bold border-2 ${
        isReal
          ? "bg-green-900/40 text-green-300 border-green-600"
          : "bg-red-900/40 text-red-300 border-red-600"
      }`}
    >
      {isReal ? "REAL MUSIC DETECTED" : "AI-GENERATED AUDIO DETECTED"}
    </div>
  );
}
