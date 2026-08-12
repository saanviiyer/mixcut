"""mixcut command-line interface.

Examples:
    python -m mixcut song.mp3 -o short.wav
    python -m mixcut song.wav --dry-run
    python -m mixcut song.wav -o short.wav --crossfade-bars 2
    python -m mixcut song.wav -o short.wav --remove 45.0-88.5
"""

from __future__ import annotations

import argparse
import json
import sys

from .analysis import analyze
from .render import render_cut, write_outputs


def _parse_span(text: str) -> tuple[float, float]:
    if "-" not in text:
        raise argparse.ArgumentTypeError("--remove must look like START-END (seconds)")
    lo, hi = text.split("-", 1)
    return float(lo), float(hi)


def _print_structure(a) -> None:
    print(f"tempo: {a.tempo:.1f} BPM   duration: {a.duration:.2f}s   "
          f"beats: {len(a.beats)}   sr: {a.sr}")
    print("sections:")
    for s in a.sections:
        mark = "  <== CHORUS" if s.is_chorus else ""
        print(f"  {s.label:8s} cl{s.cluster}  {s.start:7.2f} - {s.end:7.2f}s  "
              f"({s.end - s.start:5.2f}s, energy {s.energy:.2f}){mark}")
    print(f"proposed removal: {a.remove_start:.2f}s - {a.remove_end:.2f}s "
          f"({a.remove_end - a.remove_start:.2f}s)")
    print(f"reason: {a.remove_reason}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="mixcut",
        description="Shorten a song for DJ mixes by cutting the 2nd verse+chorus "
                    "and splicing with a beat-aligned equal-power crossfade.",
    )
    p.add_argument("input", help="input audio file (wav/aiff/mp3/...)")
    p.add_argument("-o", "--output", help="output path (writes .wav and .mp3)")
    p.add_argument("--dry-run", action="store_true",
                   help="print detected structure + proposed span, write nothing")
    p.add_argument("--crossfade-bars", type=float, default=1.0,
                   help="crossfade length in bars (default 1)")
    p.add_argument("--crossfade-seconds", type=float, default=0.0,
                   help="override crossfade length in seconds (takes priority)")
    p.add_argument("--remove", type=_parse_span, default=None,
                   metavar="START-END",
                   help="manual removal span in seconds, e.g. 45.0-88.5")
    p.add_argument("--json", action="store_true",
                   help="also print the analysis as JSON")
    args = p.parse_args(argv)

    try:
        a = analyze(args.input)
    except Exception as e:
        print(f"error: analysis failed: {e}", file=sys.stderr)
        return 2

    _print_structure(a)
    if args.json:
        print(json.dumps(a.to_dict(), indent=2))

    if args.dry_run:
        return 0

    if not args.output:
        print("\n(no -o/--output given; nothing written. Add -o out.wav to export, "
              "or --dry-run to silence this.)")
        return 0

    if args.remove is not None:
        rem_start, rem_end = args.remove
        span_note = "manual --remove span"
    else:
        rem_start, rem_end = a.remove_start, a.remove_end
        span_note = "auto-detected span"

    try:
        result = render_cut(
            args.input,
            remove_start=rem_start,
            remove_end=rem_end,
            crossfade_seconds=args.crossfade_seconds,
            crossfade_bars=args.crossfade_bars,
            tempo=a.tempo,
            beats_per_bar=a.beats_per_bar,
        )
    except Exception as e:
        print(f"error: render failed: {e}", file=sys.stderr)
        return 3

    outs = write_outputs(result, args.output)
    print(f"\nusing {span_note}: {rem_start:.2f}-{rem_end:.2f}s")
    print(f"removed {result.removed_duration:.2f}s, "
          f"crossfade {result.crossfade_seconds*1000:.0f}ms")
    print(f"output duration: {result.out_duration:.2f}s "
          f"(was {result.orig_duration:.2f}s)")
    print(f"join discontinuity (click check): {result.join_discontinuity:.5f}")
    print(f"wrote: {outs['wav']}"
          + (f" and {outs['mp3']}" if outs['mp3'] else " (mp3 skipped: no ffmpeg)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
