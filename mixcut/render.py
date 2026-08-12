"""Cut + splice. Removes [remove_start, remove_end] from the audio and joins
the two halves with an equal-power (constant-power) crossfade so the join is
click-free. Works on the original stereo data at the original sample rate."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .audio_io import load_audio, save_wav, save_mp3


@dataclass
class CutResult:
    sr: int
    n_channels: int
    orig_duration: float
    out_duration: float
    removed_duration: float
    crossfade_seconds: float
    join_sample: int          # index in the output where the splice sits
    join_discontinuity: float  # max abs sample step across the join
    data: np.ndarray          # (n_samples, n_channels) float32


def _equal_power_ramps(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Constant-power fade curves of length n. fade_out**2 + fade_in**2 == 1."""
    t = np.linspace(0.0, 1.0, n, endpoint=True)
    fade_in = np.sin(t * np.pi / 2.0)
    fade_out = np.cos(t * np.pi / 2.0)
    return fade_out, fade_in


def render_cut(
    path: str,
    remove_start: float,
    remove_end: float,
    crossfade_seconds: float = 0.0,
    tempo: float | None = None,
    crossfade_bars: float | None = None,
    beats_per_bar: int = 4,
) -> CutResult:
    """Remove [remove_start, remove_end] seconds and crossfade the two halves.

    Crossfade length priority: explicit crossfade_seconds if > 0, else
    crossfade_bars * (beats_per_bar beats) using tempo, else a 0.05s default.
    The crossfade consumes material from the tail of the pre-cut segment and
    the head of the post-cut segment, so the removed content is truly gone.
    """
    data, sr = load_audio(path)          # (n, ch) float32
    n, ch = data.shape
    orig_duration = n / sr

    a = int(round(max(0.0, remove_start) * sr))
    b = int(round(min(orig_duration, remove_end) * sr))
    a = max(0, min(a, n))
    b = max(0, min(b, n))
    if b <= a:
        raise ValueError(
            f"Empty removal span: start={remove_start:.3f}s end={remove_end:.3f}s"
        )

    # Determine crossfade length in samples.
    if crossfade_seconds and crossfade_seconds > 0:
        xf = crossfade_seconds
    elif crossfade_bars and tempo and tempo > 0:
        seconds_per_beat = 60.0 / tempo
        xf = crossfade_bars * beats_per_bar * seconds_per_beat
    else:
        xf = 0.05
    xf_n = int(round(xf * sr))
    xf_n = max(1, xf_n)

    left = data[:a]                       # keep the head
    right = data[b:]                      # keep the tail

    # Cap the crossfade to available material on both sides.
    xf_n = min(xf_n, len(left), len(right))
    if xf_n < 1:
        # No room to crossfade; hard concatenate (rare, very short clips).
        out = np.concatenate([left, right], axis=0)
        join = len(left)
        disc = _join_discontinuity(out, join)
        return CutResult(
            sr=sr, n_channels=ch, orig_duration=orig_duration,
            out_duration=len(out) / sr, removed_duration=(b - a) / sr,
            crossfade_seconds=0.0, join_sample=join,
            join_discontinuity=disc, data=out.astype(np.float32),
        )

    fade_out, fade_in = _equal_power_ramps(xf_n)
    fade_out = fade_out[:, None]
    fade_in = fade_in[:, None]

    left_body = left[: len(left) - xf_n]
    left_tail = left[len(left) - xf_n :]
    right_head = right[:xf_n]
    right_body = right[xf_n:]

    blended = left_tail * fade_out + right_head * fade_in

    out = np.concatenate([left_body, blended, right_body], axis=0)
    join = len(left_body) + xf_n // 2  # middle of the crossfade region

    disc = _join_discontinuity(out, len(left_body), xf_n)

    return CutResult(
        sr=sr,
        n_channels=ch,
        orig_duration=orig_duration,
        out_duration=len(out) / sr,
        removed_duration=(b - a) / sr,
        crossfade_seconds=xf_n / sr,
        join_sample=join,
        join_discontinuity=disc,
        data=out.astype(np.float32),
    )


def _join_discontinuity(out: np.ndarray, join: int, window: int = 1) -> float:
    """Max absolute sample-to-sample step in a small window around the join,
    used as a click detector. A clean equal-power crossfade keeps this on the
    order of ordinary within-signal steps (no impulse)."""
    lo = max(1, join - max(window, 4))
    hi = min(len(out), join + max(window, 4) + 1)
    if hi - lo < 2:
        return 0.0
    seg = out[lo:hi]
    steps = np.abs(np.diff(seg, axis=0))
    return float(steps.max())


def write_outputs(result: CutResult, out_path: str) -> dict:
    """Write wav (always) and mp3 (if ffmpeg) next to out_path.
    out_path may end in .wav or .mp3 or have no extension; both are produced."""
    base, ext = os.path.splitext(out_path)
    wav_path = base + ".wav"
    mp3_path = base + ".mp3"
    save_wav(wav_path, result.data, result.sr)
    mp3_ok = save_mp3(mp3_path, result.data, result.sr)
    return {"wav": wav_path, "mp3": mp3_path if mp3_ok else None}
