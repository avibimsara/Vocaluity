import csv
import io
import urllib.request
from pathlib import Path
import subprocess
import os
import sys
import time

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw" / "fakemusiccaps" / "real"
CSV_URL = "https://huggingface.co/datasets/google/MusicCaps/resolve/main/musiccaps-public.csv"

# ffmpeg is needed by yt-dlp for audio extraction and segment cutting
FFMPEG_DIR = None
_ffmpeg_env = os.environ.get("FFMPEG_DIR", "")
if _ffmpeg_env and Path(_ffmpeg_env).exists() and (Path(_ffmpeg_env) / "ffmpeg.exe").exists():
    FFMPEG_DIR = Path(_ffmpeg_env)
else:
    # Auto-detect winget install location
    winget_base = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    if winget_base.exists():
        for d in winget_base.iterdir():
            if "FFmpeg" in d.name:
                candidate = next(d.glob("ffmpeg-*-full_build/bin"), None)
                if candidate and (candidate / "ffmpeg.exe").exists():
                    FFMPEG_DIR = candidate
                    break


def download_csv():
    """Download the MusicCaps metadata CSV."""
    print("Downloading MusicCaps metadata...", flush=True)
    data = urllib.request.urlopen(CSV_URL).read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(data))
    rows = list(reader)
    print(f"Found {len(rows)} clips in metadata.", flush=True)
    return rows


def download_clip(row, output_dir, yt_dlp_path, env):
    """Download a single clip using yt-dlp.

    Returns (ytid, status) where status is one of:
      "ok", "skipped", "unavailable", "error: ...", "timeout"
    """
    ytid = row["ytid"]
    start = float(row["start_s"])
    end = float(row["end_s"])
    out_path = output_dir / f"{ytid}.wav"

    if out_path.exists():
        return ytid, "skipped"

    url = f"https://www.youtube.com/watch?v={ytid}"

    cmd = [
        yt_dlp_path,
        "--no-warnings",
        "-q",
        "-x",
        "--audio-format", "wav",
        "--download-sections", f"*{start}-{end}",
        "--force-keyframes-at-cuts",
        "-o", str(out_path),
        url,
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=90, env=env,
        )
        if result.returncode == 0 and out_path.exists():
            return ytid, "ok"

        stderr = result.stderr.lower()
        if "not available" in stderr or "private video" in stderr or "removed" in stderr or "sign in" in stderr:
            return ytid, "unavailable"

        return ytid, f"error: {result.stderr.strip()[-150:]}"
    except subprocess.TimeoutExpired:
        return ytid, "timeout"
    except Exception as e:
        return ytid, f"exception: {e}"


def main():
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find yt-dlp in the venv
    venv_dir = Path(__file__).parent / "venv" / "Scripts"
    yt_dlp_path = str(venv_dir / "yt-dlp.exe") if (venv_dir / "yt-dlp.exe").exists() else "yt-dlp"

    # Prepare env with ffmpeg
    env = os.environ.copy()
    if FFMPEG_DIR:
        env["PATH"] = str(FFMPEG_DIR) + ";" + env.get("PATH", "")
        print(f"Using ffmpeg from: {FFMPEG_DIR}", flush=True)
    else:
        print("WARNING: ffmpeg not found — downloads will fail!", flush=True)
        print("Install ffmpeg or set FFMPEG_DIR env var.", flush=True)
        return

    rows = download_csv()

    # Check how many already downloaded
    existing = set(f.stem for f in output_dir.glob("*.wav"))
    remaining = [r for r in rows if r["ytid"] not in existing]
    print(f"Already downloaded: {len(existing)}, remaining: {len(remaining)}", flush=True)

    if not remaining:
        print("All clips already downloaded.", flush=True)
        return

    ok = 0
    unavailable = 0
    errors = 0
    last_error = ""

    for i, row in enumerate(remaining, 1):
        ytid, status = download_clip(row, output_dir, yt_dlp_path, env)

        if status == "ok" or status == "skipped":
            ok += 1
        elif status == "unavailable":
            unavailable += 1
        else:
            errors += 1
            last_error = f"{ytid}: {status}"

        if i % 50 == 0 or i == len(remaining):
            total_ok = ok + len(existing)
            msg = (
                f"[{i}/{len(remaining)}] downloaded={total_ok} "
                f"unavailable={unavailable} errors={errors}"
            )
            if last_error:
                msg += f"  last_err: {last_error[:100]}"
            print(msg, flush=True)

    total = len(list(output_dir.glob("*.wav")))
    print(f"\nDone. {total} wav files in {output_dir}", flush=True)
    print(f"({ok} new, {len(existing)} existed, {unavailable} unavailable, {errors} errors)", flush=True)


if __name__ == "__main__":
    main()
