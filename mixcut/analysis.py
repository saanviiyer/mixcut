"""Structure analysis: beat tracking + beat-synchronous chroma/MFCC ->
recurrence (self-similarity) matrix -> Laplacian spectral clustering into
labeled sections, then a heuristic to name the repeated high-energy cluster
CHORUS and pick a default "remove 2nd verse + chorus" span.

DSP path: librosa (available on this Python 3.13 via librosa 1.0.0 + numba
0.67.0 wheels). This mirrors librosa's Laplacian segmentation reference.

Detection is approximate by nature. The whole point of mixcut is that a human
reviews and adjusts the span before export.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
import librosa
import scipy.ndimage
import scipy.linalg
import scipy.sparse.csgraph
from sklearn.cluster import KMeans

from .audio_io import load_audio, to_mono


@dataclass
class Section:
    label: str          # e.g. "chorus", "verse", "A", "B" ...
    cluster: int        # raw cluster id from spectral clustering
    start: float        # seconds
    end: float          # seconds
    is_chorus: bool
    energy: float       # mean RMS over the section (0..1-ish)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Analysis:
    duration: float
    sr: int
    tempo: float
    beats: list[float]                 # beat times (s)
    downbeats: list[float]             # estimated downbeat times (s)
    sections: list[Section]
    remove_start: float                # proposed removal span start (s)
    remove_end: float                  # proposed removal span end (s)
    remove_reason: str
    beats_per_bar: int = 4

    def to_dict(self) -> dict:
        d = {
            "duration": self.duration,
            "sr": self.sr,
            "tempo": self.tempo,
            "beats": self.beats,
            "downbeats": self.downbeats,
            "beats_per_bar": self.beats_per_bar,
            "sections": [s.to_dict() for s in self.sections],
            "remove_start": self.remove_start,
            "remove_end": self.remove_end,
            "remove_reason": self.remove_reason,
        }
        return d


def _laplacian_segmentation(
    Csync: np.ndarray, Msync: np.ndarray, k: int
) -> np.ndarray:
    """Return a per-beat integer label array using the librosa Laplacian
    segmentation recipe: combine a recurrence graph (from chroma) with a
    local sequence graph (from MFCC), take the symmetric normalized Laplacian,
    and KMeans-cluster the first k eigenvectors."""
    n = Csync.shape[1]

    # Recurrence (repetition) graph from chroma.
    R = librosa.segment.recurrence_matrix(
        Csync, width=3, mode="affinity", sym=True
    )
    # Smooth along the diagonal to enforce timbral continuity.
    df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
    Rf = df(R, size=(1, 7))

    # Local sequence graph from MFCC (path/temporal connectivity).
    path_distance = np.sum(np.diff(Msync, axis=1) ** 2, axis=0)
    sigma = np.median(path_distance) if path_distance.size else 1.0
    sigma = sigma if sigma > 0 else 1.0
    path_sim = np.exp(-path_distance / sigma)
    R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)

    # Balance the two graphs (mu weighting from the librosa example).
    deg_path = np.sum(R_path, axis=1)
    deg_rec = np.sum(Rf, axis=1)
    mu = deg_path.dot(deg_path + deg_rec)
    denom = np.sum((deg_path + deg_rec) ** 2)
    mu = mu / denom if denom > 0 else 0.5
    mu = float(np.clip(mu, 0.0, 1.0))

    A = mu * Rf + (1 - mu) * R_path

    # Symmetric normalized Laplacian.
    L = scipy.sparse.csgraph.laplacian(A, normed=True)
    evals, evecs = scipy.linalg.eigh(L)

    # Use the smoothed low eigenvectors as the embedding. The smoothing window
    # must scale with track length: the librosa reference uses 9 on songs with
    # hundreds of beats, but on a short track (tens of beats) a wide median
    # smears section boundaries and can merge/destroy repeats. Scale to ~n/8.
    ev_smooth = int(np.clip(n // 8, 3, 9))
    evecs = scipy.ndimage.median_filter(evecs, size=(ev_smooth, 1))
    k = int(np.clip(k, 2, max(2, n - 1)))
    X = evecs[:, :k]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms

    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = km.fit_predict(Xn)
    return labels


def _beats_to_sections(
    labels: np.ndarray,
    beat_times: np.ndarray,
    duration: float,
    rms_beat: np.ndarray,
) -> list[tuple[int, float, float, float]]:
    """Merge consecutive same-label beats into (cluster, start, end, energy)."""
    segs: list[tuple[int, float, float, float]] = []
    if len(labels) == 0:
        return segs
    # beat_times has one time per beat; section spans from this beat's time to
    # the next boundary. Append duration as the final edge.
    edges = np.append(beat_times, duration)
    cur = labels[0]
    seg_start_idx = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != cur:
            start = float(edges[seg_start_idx])
            end = float(edges[i])
            energy = float(np.mean(rms_beat[seg_start_idx:i])) if i > seg_start_idx else 0.0
            segs.append((int(cur), start, end, energy))
            if i < len(labels):
                cur = labels[i]
                seg_start_idx = i
    return segs


def _snap_to_beat(t: float, beats: np.ndarray, prefer: np.ndarray | None = None) -> float:
    """Snap time t to the nearest beat (or, if prefer grid given, nearest of
    that grid, e.g. downbeats)."""
    grid = prefer if (prefer is not None and len(prefer)) else beats
    if grid is None or len(grid) == 0:
        return t
    idx = int(np.argmin(np.abs(grid - t)))
    return float(grid[idx])


def analyze(
    path: str,
    hop_length: int = 512,
    max_clusters: int = 6,
    beats_per_bar: int = 4,
) -> Analysis:
    """Full structure analysis of an audio file."""
    stereo, sr = load_audio(path)
    y = to_mono(stereo)
    duration = len(y) / sr

    # --- Beat tracking ---
    tempo, beat_frames = librosa.beat.beat_track(
        y=y, sr=sr, hop_length=hop_length, trim=False
    )
    tempo = float(np.atleast_1d(tempo)[0])
    # Fold octave errors into a musical range so a "1 bar" crossfade is sane.
    # (This only rescales the reported BPM used for bar<->seconds; the beat
    # grid itself, used for boundary snapping, is unchanged.)
    if tempo > 0:
        while tempo < 90:
            tempo *= 2.0
        while tempo >= 180:
            tempo /= 2.0
    beat_frames = np.asarray(beat_frames)
    if beat_frames.size < 4:
        # Fallback: synthesize a beat grid from tempo.
        spb = 60.0 / max(tempo, 1e-6)
        beat_times = np.arange(0, duration, spb)
        beat_frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

    # Estimated downbeats: every beats_per_bar-th beat (phase 0). Coarse but
    # good enough for snapping cut boundaries to bar lines.
    downbeats = beat_times[::beats_per_bar]

    # --- Features (beat synchronous) ---
    chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    Csync = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
    Msync = librosa.util.sync(mfcc, beat_frames, aggregate=np.mean)
    rms_beat = librosa.util.sync(rms[None, :], beat_frames, aggregate=np.mean)[0]
    # Normalize energy to 0..1 for readable section energies.
    if rms_beat.max() > 0:
        rms_beat = rms_beat / rms_beat.max()

    n_beats = Csync.shape[1]
    # sync can yield one more column than beats; align lengths.
    m = min(n_beats, len(beat_times), Msync.shape[1], len(rms_beat))
    Csync = Csync[:, :m]
    Msync = Msync[:, :m]
    rms_beat = rms_beat[:m]
    beat_times_use = beat_times[:m]

    # --- Segmentation ---
    k = int(np.clip(max_clusters, 2, max(2, m - 1)))
    if m >= 4:
        labels = _laplacian_segmentation(Csync, Msync, k)
    else:
        labels = np.zeros(m, dtype=int)

    segs = _beats_to_sections(labels, beat_times_use, duration, rms_beat)

    # --- Identify chorus cluster ---
    # Group segments by cluster; the chorus is the repeated, high-energy cluster.
    from collections import defaultdict

    by_cluster: dict[int, list[tuple[int, float, float, float]]] = defaultdict(list)
    for s in segs:
        by_cluster[s[0]].append(s)

    def cluster_score(cid: int) -> tuple:
        occ = by_cluster[cid]
        count = len(occ)
        mean_energy = np.mean([o[3] for o in occ]) if occ else 0.0
        total_dur = np.sum([o[2] - o[1] for o in occ]) if occ else 0.0
        # Prefer clusters that repeat >=2 times and are energetic.
        repeats_bonus = 1 if count >= 2 else 0
        return (repeats_bonus, count * mean_energy, mean_energy, total_dur)

    chorus_cluster = None
    if by_cluster:
        candidates = [c for c in by_cluster if len(by_cluster[c]) >= 2]
        if candidates:
            chorus_cluster = max(candidates, key=cluster_score)
        else:
            # No repetition detected; fall back to the highest-energy cluster.
            chorus_cluster = max(by_cluster, key=lambda c: np.mean([o[3] for o in by_cluster[c]]))

    # --- Label sections ---
    # Assign human-ish labels: chorus cluster -> "chorus"; the most common
    # non-chorus cluster -> "verse"; the rest keep letter labels.
    letters = "ABCDEFGH"
    cluster_letter: dict[int, str] = {}
    ordered = sorted(by_cluster, key=lambda c: -len(by_cluster[c]))
    li = 0
    for c in ordered:
        cluster_letter[c] = letters[li % len(letters)]
        li += 1

    # Which cluster looks like the verse: most frequent non-chorus cluster.
    verse_cluster = None
    non_chorus = [c for c in by_cluster if c != chorus_cluster]
    if non_chorus:
        verse_cluster = max(non_chorus, key=lambda c: len(by_cluster[c]))

    sections: list[Section] = []
    for (cid, start, end, energy) in segs:
        is_chorus = cid == chorus_cluster
        if is_chorus:
            label = "chorus"
        elif cid == verse_cluster:
            label = "verse"
        else:
            label = cluster_letter.get(cid, "?")
        sections.append(
            Section(
                label=label,
                cluster=cid,
                start=round(start, 3),
                end=round(end, 3),
                is_chorus=is_chorus,
                energy=round(float(energy), 3),
            )
        )

    # --- Choose default removal span: 2nd verse + 2nd chorus ---
    remove_start, remove_end, reason = _choose_removal(
        sections, chorus_cluster, verse_cluster, beat_times_use, downbeats, duration
    )

    return Analysis(
        duration=round(duration, 3),
        sr=sr,
        tempo=round(tempo, 2),
        beats=[round(float(t), 3) for t in beat_times_use],
        downbeats=[round(float(t), 3) for t in downbeats],
        sections=sections,
        remove_start=round(remove_start, 3),
        remove_end=round(remove_end, 3),
        remove_reason=reason,
        beats_per_bar=beats_per_bar,
    )


def _choose_removal(
    sections: list[Section],
    chorus_cluster: int | None,
    verse_cluster: int | None,
    beats: np.ndarray,
    downbeats: np.ndarray,
    duration: float,
) -> tuple[float, float, str]:
    """Default span = from the start of the 2nd verse (== end of the 1st
    chorus) through the end of the 2nd chorus. Snap to bar/downbeat lines.

    Falls back gracefully when structure is ambiguous."""
    chorus_idx = [i for i, s in enumerate(sections) if s.is_chorus]

    if len(chorus_idx) >= 2:
        first_chorus = chorus_idx[0]
        second_chorus = chorus_idx[1]
        # Removal starts right after the first chorus ends (start of 2nd verse)
        start = sections[first_chorus].end
        # Removal ends at the end of the 2nd chorus.
        end = sections[second_chorus].end
        start = _snap_to_beat(start, beats, downbeats)
        end = _snap_to_beat(end, beats, downbeats)
        if end <= start:
            end = _snap_to_beat(sections[second_chorus].end, beats, None)
        reason = (
            "Removing from end of 1st chorus (start of 2nd verse) "
            "through end of 2nd chorus."
        )
        return start, end, reason

    if len(chorus_idx) == 1 and len(sections) > chorus_idx[0] + 1:
        # Only one chorus found: remove the single chorus + the following
        # section as a best-effort default.
        c = chorus_idx[0]
        start = _snap_to_beat(sections[c].start, beats, downbeats)
        nxt = min(c + 1, len(sections) - 1)
        end = _snap_to_beat(sections[nxt].end, beats, downbeats)
        return start, end, "Only one chorus detected; proposing chorus + next section (please review)."

    # Fallback: remove the middle third.
    start = _snap_to_beat(duration / 3.0, beats, downbeats)
    end = _snap_to_beat(2.0 * duration / 3.0, beats, downbeats)
    return start, end, "Structure ambiguous; proposing the middle third (please review)."
