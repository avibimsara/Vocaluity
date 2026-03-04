# download_suno.py - Download Suno AI-generated clips for Vocaluity training
#
# Source:  nyuuzyou/suno on HuggingFace  (CC0 licence)
# Output:  data/raw/fakemusiccaps/suno/*.wav
#
# Uses HTTP range requests to fetch only the first ~600 KB of each MP3
# (enough for 10 s at any bitrate) rather than downloading full tracks.
# Estimated final size: ~2.1 GB for 5,000 clips.

import os
import sys
import time
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TARGET_CLIPS  = 5000       # number of clips to download
CLIP_DURATION = 10         # seconds  — matches DURATION in config.py
SAMPLE_RATE   = 22050      # Hz       — matches config.py
RANGE_BYTES   = 600_000    # 600 KB   — enough for 10 s at 320 kbps + headers
REQUEST_DELAY = 0.2        # seconds between requests  (be polite to the CDN)

OUTPUT_DIR = (
    Path(__file__).parent.parent / "data" / "raw" / "fakemusiccaps" / "suno"
)


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def try_download_clip(audio_url: str, output_path: Path) -> bool:
    """
    Fetch the first RANGE_BYTES bytes of audio_url, decode with librosa,
    trim/pad to exactly CLIP_DURATION seconds, and write as a mono WAV.

    Returns True on success, False on any failure (silently — caller tracks
    statistics).
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; academic-research-bot)",
            "Range": f"bytes=0-{RANGE_BYTES - 1}",
        }
        resp = requests.get(audio_url, headers=headers, timeout=30)

        # Accept 206 Partial Content or 200 OK (CDN ignores range header)
        if resp.status_code not in (200, 206):
            return False

        # Sanity-check: skip suspiciously small responses (errors / empty)
        if len(resp.content) < 10_000:
            return False

        # Write raw bytes to a temp file so librosa can decode the codec
        suffix = ".mp3" if audio_url.lower().endswith(".mp3") else ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name

        try:
            y, _ = librosa.load(
                tmp_path,
                sr=SAMPLE_RATE,
                duration=CLIP_DURATION,
                mono=True,
            )

            # Skip clips where less than 3 s decoded (corrupt / near-silent)
            if len(y) < SAMPLE_RATE * 3:
                return False

            # Trim or zero-pad to exactly CLIP_DURATION seconds
            target_len = SAMPLE_RATE * CLIP_DURATION
            if len(y) < target_len:
                y = np.pad(y, (0, target_len - len(y)), mode="constant")
            else:
                y = y[:target_len]

            sf.write(str(output_path), y, SAMPLE_RATE, subtype="PCM_16")
            return True

        finally:
            os.unlink(tmp_path)

    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Resume support: count files already downloaded
    existing = {f.stem for f in OUTPUT_DIR.glob("*.wav")}
    remaining = TARGET_CLIPS - len(existing)

    if remaining <= 0:
        print(f"Already have {len(existing)} Suno clips. Nothing to do.")
        return

    print("=" * 60)
    print("Suno AI clip downloader")
    print("=" * 60)
    print(f"Target clips  : {TARGET_CLIPS}")
    print(f"Already saved : {len(existing)}")
    print(f"To download   : {remaining}")
    print(f"Clip length   : {CLIP_DURATION}s  ({SAMPLE_RATE} Hz mono WAV)")
    print(f"Estimated size: ~{remaining * 0.43:.0f} MB new  "
          f"/ ~{TARGET_CLIPS * 0.43:.0f} MB total")
    print(f"Output dir    : {OUTPUT_DIR}")
    print()

    # Load HuggingFace dataset in streaming mode (avoids pulling all metadata
    # into memory — only iterates one record at a time)
    print("Connecting to HuggingFace dataset (nyuuzyou/suno)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("nyuuzyou/suno", split="train", streaming=True)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Make sure 'datasets' is installed:  pip install datasets")
        return

    downloaded = 0
    failed     = 0
    skipped    = 0

    with tqdm(total=remaining, unit="clip", desc="Downloading") as pbar:
        for sample in ds:
            if downloaded >= remaining:
                break

            # --- Quality filters -------------------------------------------
            # Skip private clips
            if not sample.get("is_public", True):
                skipped += 1
                continue

            # Skip clips that errored during generation
            if sample.get("metadata_error_type"):
                skipped += 1
                continue

            # Skip clips shorter than 15 s (can't get a clean 10 s segment)
            duration = sample.get("metadata_duration")
            if duration is not None and float(duration) < 15:
                skipped += 1
                continue

            # Must have a valid audio URL
            audio_url = sample.get("audio_url", "")
            if not audio_url or not audio_url.startswith("http"):
                skipped += 1
                continue

            # --- Build output filename from song ID ------------------------
            clip_id    = sample.get("id", f"{len(existing) + downloaded:06d}")
            fname      = f"suno_{clip_id}.wav"
            output_path = OUTPUT_DIR / fname

            # Skip if already downloaded in a previous run
            if fname[:-4] in existing:
                skipped += 1
                continue

            # --- Download --------------------------------------------------
            if try_download_clip(audio_url, output_path):
                downloaded += 1
                pbar.update(1)
                pbar.set_postfix({"ok": downloaded, "fail": failed})
            else:
                failed += 1

            time.sleep(REQUEST_DELAY)

    # --- Summary -----------------------------------------------------------
    total   = len(list(OUTPUT_DIR.glob("*.wav")))
    size_gb = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*.wav")) / 1e9

    print()
    print("=" * 60)
    print(f"Download complete: {total} WAV files in {OUTPUT_DIR}")
    print(f"  New this run : {downloaded}")
    print(f"  Pre-existing : {len(existing)}")
    print(f"  Failed/skip  : {failed + skipped}")
    print(f"  Folder size  : {size_gb:.2f} GB")
    print()
    if total < TARGET_CLIPS:
        print(f"NOTE: Only got {total}/{TARGET_CLIPS} clips — some CDN URLs may")
        print("have expired. Re-run the script to attempt more downloads.")
    else:
        print("Ready to retrain. Run:  python train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
