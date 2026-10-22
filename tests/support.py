import math
import struct
import wave
from pathlib import Path


def write_tone(path: Path, seconds: float = 5.0, rate: int = 16000) -> None:
    count = int(seconds * rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        frames = bytearray()
        for index in range(count):
            sample = int(0.2 * 32767 * math.sin(2 * math.pi * 440 * index / rate))
            frames += struct.pack("<h", sample)
        handle.writeframes(frames)
