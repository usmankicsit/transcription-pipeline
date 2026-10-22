import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
tools = ROOT / ".tools"
ffmpeg = tools / "ffmpeg"
ffprobe = tools / "ffprobe"
if ffmpeg.is_file() and ffprobe.is_file():
    os.environ.setdefault("FFMPEG_BINARY", str(ffmpeg))
    os.environ.setdefault("FFPROBE_BINARY", str(ffprobe))
