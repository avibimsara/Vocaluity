"use client";

import { useState } from "react";

interface Props {
  showVisualizations: boolean;
  onToggleVisualizations: (v: boolean) => void;
  showTechnical: boolean;
  onToggleTechnical: (v: boolean) => void;
}

export default function Navbar({
  showVisualizations,
  onToggleVisualizations,
  showTechnical,
  onToggleTechnical,
}: Props) {
  const [settingsOpen, setSettingsOpen] = useState(false);

  return (
    <nav className="sticky top-0 z-50 bg-gray-900/80 backdrop-blur border-b border-gray-800">
      <div className="max-w-6xl mx-auto flex items-center justify-between px-4 h-14">
        {/* Branding */}
        <div className="flex items-center gap-2">
          <span className="text-lg font-bold text-blue-400">Vocaluity</span>
          <span className="hidden sm:inline text-xs text-gray-500">
            AI-Generated Audio Detection
          </span>
        </div>

        {/* Desktop toggles */}
        <div className="hidden md:flex items-center gap-5">
          <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={showVisualizations}
              onChange={(e) => onToggleVisualizations(e.target.checked)}
              className="accent-blue-500"
            />
            Visualizations
          </label>
          <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={showTechnical}
              onChange={(e) => onToggleTechnical(e.target.checked)}
              className="accent-blue-500"
            />
            Technical Details
          </label>
        </div>

        {/* Mobile gear button */}
        <button
          className="md:hidden p-2 text-gray-400 hover:text-gray-200"
          onClick={() => setSettingsOpen((o) => !o)}
          aria-label="Toggle settings"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-5 w-5"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
            />
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
            />
          </svg>
        </button>
      </div>

      {/* Mobile dropdown */}
      {settingsOpen && (
        <div className="md:hidden border-t border-gray-800 bg-gray-900 px-4 py-3 space-y-3">
          <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={showVisualizations}
              onChange={(e) => onToggleVisualizations(e.target.checked)}
              className="accent-blue-500"
            />
            Show Visualizations
          </label>
          <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={showTechnical}
              onChange={(e) => onToggleTechnical(e.target.checked)}
              className="accent-blue-500"
            />
            Show Technical Details
          </label>
        </div>
      )}
    </nav>
  );
}
