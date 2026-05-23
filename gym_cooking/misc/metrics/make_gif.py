"""Turn a directory of PNG frames into a GIF.

Usage:
    python -m misc.metrics.make_gif --frames misc/game/record/<runname> \
                                    --out ../images_svo/<file>.gif \
                                    --duration 200
"""
import argparse
import os
import sys
from glob import glob

from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", required=True, help="Directory of PNG frames.")
    p.add_argument("--out", required=True, help="Output GIF path.")
    p.add_argument("--duration", type=int, default=200,
                   help="Per-frame duration in ms.")
    p.add_argument("--loop", type=int, default=0, help="GIF loop count (0 = forever).")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Optional cap on the number of frames included.")
    return p.parse_args()


def main():
    args = parse_args()
    paths = sorted(glob(os.path.join(args.frames, "*.png")))
    if not paths:
        sys.exit("No PNGs in {}".format(args.frames))
    if args.max_frames:
        paths = paths[: args.max_frames]
    frames = [Image.open(p).convert("RGB") for p in paths]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    frames[0].save(
        args.out,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration,
        loop=args.loop,
        optimize=True,
    )
    print("Wrote {} ({} frames, {} ms each)".format(args.out, len(frames), args.duration))


if __name__ == "__main__":
    main()
