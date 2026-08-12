"""Audio loading/saving. Uses soundfile for wav/aiff/flac; falls back to
ffmpeg for formats soundfile can't decode (e.g. mp3). Keeps the original
stereo layout and sample rate so exports match the source."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import numpy as np
import soundfile as sf

FFMPEG = os.environ.get("MIXCUT_FFMPEG", "/usr/local/bin/ffmpeg")
if not os.path.exists(FFMPEG):
    _which = shutil.which("ffmpeg")
    if _which:
        FFMPEG = _which


def have_ffmpeg() -> bool:
    return bool(FFMPEG) and (os.path.exists(FFMPEG) or shutil.which(FFMPEG) is not None)


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load an audio file.

    Returns (data, sr) where data is float32 shaped (n_samples, n_channels).
    Tries soundfile first; on failure decodes to a temp wav via ffmpeg.
    """
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        return data, sr
    except Exception:
        pass

    if not have_ffmpeg():
        raise RuntimeError(
            f"soundfile could not read {path!r} and ffmpeg is unavailable "
            f"(looked for {FFMPEG!r})."
        )

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        cmd = [FFMPEG, "-y", "-i", path, "-c:a", "pcm_s16le", tmp.name]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed to decode {path!r}:\n{proc.stderr[-800:]}"
            )
        data, sr = sf.read(tmp.name, dtype="float32", always_2d=True)
        return data, sr
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def to_mono(data: np.ndarray) -> np.ndarray:
    """Downmix (n_samples, n_channels) -> (n_samples,) mono float32."""
    if data.ndim == 1:
        return data.astype(np.float32)
    return data.mean(axis=1).astype(np.float32)


def save_wav(path: str, data: np.ndarray, sr: int) -> None:
    """Write a wav preserving channel layout. data is (n_samples, n_channels)."""
    if data.ndim == 1:
        data = data[:, None]
    sf.write(path, data, sr, subtype="PCM_16")


def save_mp3(path: str, data: np.ndarray, sr: int, bitrate: str = "320k") -> bool:
    """Encode an mp3 via ffmpeg. Returns True on success, False if ffmpeg
    is unavailable. Writes a temp wav then transcodes."""
    if not have_ffmpeg():
        return False
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        save_wav(tmp.name, data, sr)
        cmd = [FFMPEG, "-y", "-i", tmp.name, "-b:a", bitrate, path]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg mp3 encode failed:\n{proc.stderr[-800:]}")
        return True
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
