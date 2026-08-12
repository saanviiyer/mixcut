"""Generate a synthetic, structured stereo test track for verification.

Layout: intro / verse1 / chorus1 / verse2 / chorus2 / outro at a fixed tempo.
chorus1 and chorus2 use the SAME chord/melody material (so repetition is
detectable); verse1 and verse2 differ. No copyrighted audio is used.
"""

from __future__ import annotations

import numpy as np

from .audio_io import save_wav


def _note(freq: float, dur: float, sr: int, harmonics=(1.0, 0.5, 0.25)) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    wave = np.zeros_like(t)
    for i, amp in enumerate(harmonics, start=1):
        wave += amp * np.sin(2 * np.pi * freq * i * t)
    # Short attack/decay envelope so individual notes don't click.
    env = np.ones_like(t)
    a = max(1, int(0.01 * sr))
    env[:a] = np.linspace(0, 1, a)
    env[-a:] = np.linspace(1, 0, a)
    return (wave * env).astype(np.float32)


def _section(chords, beats, tempo, sr, energy=1.0, kick=True) -> np.ndarray:
    """Build a section from a chord progression, one chord per bar (4 beats),
    with a simple kick on each beat to give onset structure for beat tracking."""
    spb = 60.0 / tempo
    bar = 4
    out = []
    beats_done = 0
    ci = 0
    while beats_done < beats:
        chord = chords[ci % len(chords)]
        ci += 1
        # Chord: sustained triad over one bar.
        bar_len = int(spb * bar * sr)
        seg = np.zeros(bar_len, dtype=np.float32)
        for f in chord:
            note = _note(f, spb * bar, sr)
            seg[: len(note)] += note[: len(seg)]
        seg *= energy / max(1, len(chord))
        # Kicks on each beat.
        if kick:
            for bt in range(bar):
                idx = int(bt * spb * sr)
                klen = int(0.08 * sr)
                kt = np.arange(klen) / sr
                kfreq = 90 * np.exp(-kt * 25)
                kick_wave = 0.6 * np.sin(2 * np.pi * kfreq * kt) * np.exp(-kt * 18)
                if idx + klen <= len(seg):
                    seg[idx : idx + klen] += kick_wave.astype(np.float32)
        out.append(seg)
        beats_done += bar
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def make_fixture(path: str, sr: int = 22050, tempo: float = 120.0) -> dict:
    """Write a structured stereo wav and return the ground-truth layout."""
    # Chord banks (Hz). Verses differ; choruses identical.
    verse1 = [[220.0, 277.18, 329.63], [246.94, 311.13, 369.99]]   # A, B minor-ish
    verse2 = [[196.0, 246.94, 293.66], [174.61, 220.0, 261.63]]    # G, F
    chorus = [[261.63, 329.63, 392.0], [293.66, 369.99, 440.0]]    # C, D (bright/high)
    intro = [[130.81, 164.81, 196.0]]                              # low C
    outro = [[130.81, 164.81, 196.0]]

    parts = [
        ("intro", intro, 8, 0.35, False),
        ("verse1", verse1, 16, 0.6, True),
        ("chorus1", chorus, 16, 1.0, True),
        ("verse2", verse2, 16, 0.6, True),
        ("chorus2", chorus, 16, 1.0, True),
        ("outro", outro, 8, 0.3, False),
    ]

    layout = []
    chunks = []
    t = 0.0
    for name, chords, beats, energy, kick in parts:
        seg = _section(chords, beats, tempo, sr, energy=energy, kick=kick)
        dur = len(seg) / sr
        layout.append({"name": name, "start": round(t, 3), "end": round(t + dur, 3)})
        chunks.append(seg)
        t += dur

    mono = np.concatenate(chunks)
    # Gentle normalize.
    peak = np.max(np.abs(mono)) or 1.0
    mono = 0.9 * mono / peak
    # Fake stereo: tiny delay/gain difference between channels.
    left = mono
    right = np.concatenate([np.zeros(int(0.0005 * sr), dtype=np.float32), mono])[: len(mono)]
    stereo = np.stack([left, 0.95 * right], axis=1).astype(np.float32)

    save_wav(path, stereo, sr)
    return {"path": path, "sr": sr, "tempo": tempo, "duration": round(len(mono) / sr, 3),
            "layout": layout}


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "fixture.wav"
    info = make_fixture(out)
    print(info)
