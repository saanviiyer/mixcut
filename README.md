# mixcut

Automatically shorten a song for DJ mixes by cutting the second verse + chorus, then splicing the two ends back together beat-aligned with an equal-power crossfade so the join is click-free.

---

## What it does

Long radio edits are awkward to mix. `mixcut` detects a track's structure, proposes removing the **2nd verse + 2nd chorus** (a very common way to trim a song for a set), and splices the remaining halves with a **beat-aligned, equal-power (constant-power) crossfade** so there's no click or level dip at the join.

It is **detect → review → preview → export**, never a blind cut:

1. **Detect** beat grid + section structure.
2. **Show** you the sections on a timeline with the proposed removal span highlighted.
3. **Adjust** the removal boundaries (drag handles or type seconds) and the crossfade length.
4. **Preview** the before/after in the browser.
5. **Export** WAV + MP3.

### Honest note on accuracy

**Automatic music-structure detection is approximate.** It is unsupervised clustering of audio features — it does not "understand" song form, and it will sometimes mislabel a bridge, split a section, or pick the wrong repeat, especially on tracks with unusual arrangements, key changes, or heavy production. That is exactly why the **review + preview step is not optional**: you are expected to eyeball the detected sections, nudge the removal span to land on the right boundaries, and listen to the preview before exporting. Treat the auto-detected span as a starting suggestion, not a finished decision.

---

## DSP path used: librosa

The task allowed either using `librosa` or hand-rolling the DSP in numpy/scipy if librosa/numba lacked Python 3.13 wheels. **On this machine librosa installed and imports cleanly on CPython 3.13**, because current `librosa 1.0.0` pulls `numba 0.67.0` / `llvmlite 0.49.0`, which ship 3.13 wheels. So mixcut uses librosa directly for beat tracking, CENS chroma, MFCC, and the recurrence/self-similarity matrix, and `scipy` + `scikit-learn` for the Laplacian spectral clustering. No torch/demucs or any large ML model is used — the whole stack is lightweight.

The segmentation follows librosa's **Laplacian segmentation** recipe (combine a chroma recurrence graph with an MFCC sequence graph, take the symmetric-normalized Laplacian, KMeans-cluster the low eigenvectors), with one adaptation: the eigenvector smoothing window is **scaled to the number of beats** (`~n/8`, clamped to 3–9) instead of the reference's fixed `9`. On short tracks a wide fixed window smears boundaries and destroys the very repeats we need to detect; scaling it keeps `chorus1` and `chorus2` in the same cluster. The reported BPM is also folded into a musical range (90–180) so a "1 bar" crossfade is a sane length even when the beat tracker locks onto a half/double-time octave.

---

## Install

Requires Python 3.13 and `ffmpeg` (used for MP3 encode and as a decode fallback; expected at `/usr/local/bin/ffmpeg`, override with `MIXCUT_FFMPEG`).

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

---

## CLI usage

Run as a module (`python -m mixcut`) from the project root with the venv active.

```bash
# Default action: remove the 2nd verse + chorus, write short.wav AND short.mp3
python -m mixcut song.mp3 -o short.wav

# Just inspect: print the detected structure + proposed removal span, write nothing
python -m mixcut song.wav --dry-run

# Longer crossfade (2 bars)
python -m mixcut song.wav -o short.wav --crossfade-bars 2

# Override the span manually (seconds): remove 45.0s .. 88.5s
python -m mixcut song.wav -o short.wav --remove 45.0-88.5

# Fixed crossfade length in seconds (takes priority over --crossfade-bars)
python -m mixcut song.wav -o short.wav --crossfade-seconds 1.5
```

`-o` writes both a `.wav` and a `.mp3` (the MP3 is skipped with a note if ffmpeg is unavailable). Add `--json` to also dump the full analysis (sections, beats, tempo, removal span) as JSON.

---

## Web usage

```bash
. .venv/bin/activate
uvicorn mixcut.web:app --reload
```

Then open **http://127.0.0.1:8000** (uvicorn's default host/port). Upload a song (wav/aiff/flac/mp3, ≤40 MB), and the page shows the detected sections as colored blocks (chorus / verse / other) with the proposed removal span highlighted in yellow. Drag the handles or type the start/end seconds, set the crossfade in bars, hit **Render preview**, listen to the before/after `<audio>` players, then download the WAV or MP3.

Uploads and rendered outputs are stored in a per-job work directory (default: your system temp dir under `mixcut_work/`; override with `MIXCUT_WORK`). Upload size is capped at 40 MB.

---

## Verify (synthetic fixture, no copyrighted audio)

`verify.py` generates a synthetic structured stereo track (intro / verse1 / chorus1 / verse2 / chorus2 / outro at a fixed tempo, where chorus1 and chorus2 are the same material) and runs the whole pipeline end-to-end:

```bash
. .venv/bin/activate
python verify.py
```

It checks that (a) the repeated chorus is detected, (b) the proposed removal span covers the 2nd verse + chorus, (c) the export is shorter by ~the removed span (minus the crossfade overlap), (d) the splice is click-free (no outlier sample step at the join), and that the FastAPI `/analyze` and `/render` endpoints respond. All checks pass on the fixture.

---

## Limitations

- **Detection is unsupervised and approximate** — always review and preview before exporting (see the honest note above).
- The "chorus" label is a heuristic (the repeated, high-energy cluster); on tracks where the most-repeated energetic section isn't the chorus, it can pick wrong.
- Downbeat estimation is coarse (every Nth beat from a fixed phase), not a true trained downbeat tracker.
- The default target is specifically "remove the 2nd verse + chorus"; other edit shapes require the manual `--remove` override or dragging the span.
- Very short clips may not leave room for the requested crossfade; it is capped to the available material (and falls back to a hard join only in the extreme).
- MP3 output and non-wav/aiff/flac input decoding require `ffmpeg`.
