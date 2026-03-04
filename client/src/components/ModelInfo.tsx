"use client";

import type { ModelInfo as ModelInfoType } from "@/lib/api";

interface Props {
  info: ModelInfoType | null;
}

export default function ModelInfo({ info }: Props) {
  if (!info) return null;

  return (
    <div className="space-y-1 text-sm">
      <p className="text-gray-400">
        <span className="text-gray-500">Model:</span>{" "}
        <span className="text-gray-300 break-all">{info.model_name}</span>
      </p>
      {info.accuracy > 0 && (
        <p className="text-gray-400">
          <span className="text-gray-500">Val accuracy:</span>{" "}
          <span className="text-gray-300">{(info.accuracy * 100).toFixed(2)}%</span>
        </p>
      )}
    </div>
  );
}
