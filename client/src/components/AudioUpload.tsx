"use client";

import { useCallback, useRef } from "react";

const ACCEPTED = ".wav,.mp3,.flac,.ogg,.m4a";

interface Props {
  onFileSelected: (file: File) => void;
  disabled?: boolean;
}

export default function AudioUpload({ onFileSelected, disabled }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      if (disabled) return;
      const file = e.dataTransfer.files[0];
      if (file) onFileSelected(file);
    },
    [onFileSelected, disabled],
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onFileSelected(file);
    },
    [onFileSelected],
  );

  return (
    <div
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-colors ${
        disabled
          ? "border-gray-700 text-gray-600 cursor-not-allowed"
          : "border-gray-600 hover:border-blue-500 text-gray-400 hover:text-blue-400"
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED}
        onChange={handleChange}
        className="hidden"
        disabled={disabled}
      />
      <p className="font-medium text-lg">Drop an audio file here or click to browse</p>
      <p className="text-sm mt-2 text-gray-500">WAV, MP3, FLAC, OGG, M4A</p>
    </div>
  );
}
