"""End-to-end verification on a synthetic fixture. No copyrighted audio.

Checks:
  (a) the repeated chorus sections are detected (>=2 chorus sections),
  (b) the proposed removal span covers the 2nd verse + 2nd chorus region,
  (c) the exported track is shorter by ~the removed span (minus crossfade),
  (d) the splice is click-free (small max sample step at the join),
  plus a smoke test of the FastAPI /analyze endpoint via TestClient.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

from mixcut.fixture import make_fixture
from mixcut.analysis import analyze
from mixcut.render import render_cut, write_outputs
from mixcut import audio_io


def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def main() -> int:
    work = tempfile.mkdtemp(prefix="mixcut_verify_")
    fix = os.path.join(work, "fixture.wav")
    info = make_fixture(fix)
    print("== fixture ==")
    print(f"  duration {info['duration']}s, tempo {info['tempo']} BPM")
    gt = {p["name"]: (p["start"], p["end"]) for p in info["layout"]}
    for p in info["layout"]:
        print(f"    {p['name']:8s} {p['start']:6.2f}-{p['end']:6.2f}s")

    print("\n== analysis ==")
    a = analyze(fix)
    print(f"  detected tempo {a.tempo:.1f} BPM, {len(a.sections)} sections")
    for s in a.sections:
        tag = " CHORUS" if s.is_chorus else ""
        print(f"    {s.label:8s} cl{s.cluster} {s.start:6.2f}-{s.end:6.2f}s "
              f"e{s.energy:.2f}{tag}")

    ok = True

    # (a) repeated chorus detected
    chorus_secs = [s for s in a.sections if s.is_chorus]
    a_ok = len(chorus_secs) >= 2
    print(f"\n(a) repeated chorus detected: {len(chorus_secs)} chorus sections -> "
          f"{'PASS' if a_ok else 'FAIL'}")
    ok &= a_ok

    # (b) removal span covers 2nd verse + 2nd chorus region
    tgt0 = gt["verse2"][0]
    tgt1 = gt["chorus2"][1]
    span0, span1 = a.remove_start, a.remove_end
    ov = _overlap(span0, span1, tgt0, tgt1)
    tgt_len = tgt1 - tgt0
    cover = ov / tgt_len if tgt_len else 0.0
    # also how much of the proposed span falls inside the target (precision)
    span_len = span1 - span0
    precision = ov / span_len if span_len else 0.0
    b_ok = cover >= 0.7 and precision >= 0.7
    print(f"(b) proposed removal {span0:.2f}-{span1:.2f}s vs target "
          f"verse2+chorus2 {tgt0:.2f}-{tgt1:.2f}s: coverage {cover*100:.0f}%, "
          f"precision {precision*100:.0f}% -> {'PASS' if b_ok else 'FAIL'}")
    ok &= b_ok

    # (c) exported track shorter by ~removed span (minus crossfade)
    result = render_cut(fix, remove_start=span0, remove_end=span1,
                        crossfade_bars=1.0, tempo=a.tempo,
                        beats_per_bar=a.beats_per_bar)
    # The crossfade OVERLAPS the two kept ends, so the output loses the removed
    # span plus (one) crossfade length: out = orig - removed - crossfade.
    expected_out = info["duration"] - (span1 - span0) - result.crossfade_seconds
    diff = abs(result.out_duration - expected_out)
    c_ok = result.out_duration < info["duration"] - 1.0 and diff < 0.25
    print(f"(c) duration: original {info['duration']:.2f}s -> output "
          f"{result.out_duration:.2f}s (removed {result.removed_duration:.2f}s, "
          f"crossfade {result.crossfade_seconds*1000:.0f}ms); expected "
          f"~{expected_out:.2f}s, diff {diff:.2f}s -> {'PASS' if c_ok else 'FAIL'}")
    ok &= c_ok

    # (d) click-free join. Compare the join step to the track's own typical and
    # max sample steps; the join must not introduce an outlier impulse.
    steps = np.abs(np.diff(result.data, axis=0))
    typical = float(np.percentile(steps, 99))
    global_max = float(steps.max())
    join_step = result.join_discontinuity
    d_ok = join_step <= max(3 * typical, 0.02) and join_step <= global_max + 1e-6
    print(f"(d) click-free: join step {join_step:.5f} vs 99th-pct step "
          f"{typical:.5f}, global-max step {global_max:.5f} -> "
          f"{'PASS' if d_ok else 'FAIL'}")
    ok &= d_ok

    outs = write_outputs(result, os.path.join(work, "out"))
    print(f"    wrote {outs['wav']}"
          + (f" and {outs['mp3']}" if outs["mp3"] else " (no mp3: ffmpeg missing)"))

    # FastAPI smoke test
    print("\n== web smoke test ==")
    try:
        from fastapi.testclient import TestClient
        from mixcut.web import app
        client = TestClient(app)
        with open(fix, "rb") as fh:
            r = client.post("/analyze",
                            files={"file": ("fixture.wav", fh, "audio/wav")})
        w_ok = r.status_code == 200 and "sections" in r.json()
        j = r.json() if r.status_code == 200 else {}
        print(f"  POST /analyze -> {r.status_code}, "
              f"{len(j.get('sections', []))} sections, job {j.get('job_id')} -> "
              f"{'PASS' if w_ok else 'FAIL'}")
        if w_ok:
            rr = client.post("/render", data={
                "job_id": j["job_id"],
                "remove_start": j["remove_start"],
                "remove_end": j["remove_end"],
                "crossfade_bars": 1.0,
            })
            r_ok = rr.status_code == 200 and rr.json().get("out_duration", 1e9) < info["duration"]
            print(f"  POST /render -> {rr.status_code}, out "
                  f"{rr.json().get('out_duration')}s -> {'PASS' if r_ok else 'FAIL'}")
            w_ok &= r_ok
        ok &= w_ok
    except Exception as e:
        print(f"  web smoke test FAILED: {e}")
        ok = False

    print("\n==== " + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED") + " ====")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
