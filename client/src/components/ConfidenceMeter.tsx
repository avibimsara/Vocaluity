"use client";

import type { PredictionResult } from "@/lib/api";

interface Props {
  result: PredictionResult;
  showTechnical?: boolean;
}

export default function ConfidenceMeter({ result, showTechnical }: Props) {
  const pct = (result.confidence * 100).toFixed(1);
  const realPct = ((result.probabilities.real ?? 0) * 100).toFixed(1);
  const aiPct = ((result.probabilities.ai_generated ?? 0) * 100).toFixed(1);

  return (
    <div className="space-y-5">
      {/* Main confidence */}
      <div>
        <p className="text-gray-300 font-semibold mb-1">Confidence: {pct}%</p>
        <div className="w-full h-4 bg-gray-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-blue-500 rounded-full transition-all duration-700"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      {/* Probability breakdown */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="text-sm text-gray-400 mb-1">Real Music</p>
          <p className="text-2xl font-bold text-green-400">{realPct}%</p>
          <div className="w-full h-2 bg-gray-700 rounded-full overflow-hidden mt-1">
            <div
              className="h-full bg-green-500 rounded-full transition-all duration-700"
              style={{ width: `${realPct}%` }}
            />
          </div>
        </div>
        <div>
          <p className="text-sm text-gray-400 mb-1">AI-Generated</p>
          <p className="text-2xl font-bold text-red-400">{aiPct}%</p>
          <div className="w-full h-2 bg-gray-700 rounded-full overflow-hidden mt-1">
            <div
              className="h-full bg-red-500 rounded-full transition-all duration-700"
              style={{ width: `${aiPct}%` }}
            />
          </div>
        </div>
      </div>

      {/* Technical details */}
      {showTechnical && (
        <div className="grid grid-cols-2 gap-4 pt-3 border-t border-gray-700">
          <div>
            <p className="text-xs text-gray-500">Predicted Class</p>
            <p className="text-gray-300 font-mono">{result.prediction}</p>
          </div>
          <div>
            <p className="text-xs text-gray-500">Raw Confidence</p>
            <p className="text-gray-300 font-mono">{result.confidence.toFixed(4)}</p>
          </div>
        </div>
      )}
    </div>
  );
}
