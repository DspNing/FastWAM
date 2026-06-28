#!/usr/bin/env python3
"""Show VAE latent caching progress with a live progress bar.

Usage:
    python scripts/vae_cache_progress.py                 # show once
    python scripts/vae_cache_progress.py --watch          # refresh every 10s
    python scripts/vae_cache_progress.py --watch 5        # refresh every 5s

Reads the count of latent_*.pt files in the cache dir and compares to the total
dataset size (277713). Estimates ETA from the rate over the last refresh interval.
"""
import argparse
import os
import time
from pathlib import Path

DEFAULT_CACHE_DIR = "data/vae_latents_cache/libero_wan21_112x224"
TOTAL = 277713  # LIBERO 4-suite total samples


def count_done(cache_dir: Path) -> int:
    if not cache_dir.exists():
        return 0
    # Count latent_*.pt files. os.scandir is faster than glob for large dirs.
    n = 0
    try:
        for entry in os.scandir(cache_dir):
            if entry.name.startswith("latent_") and entry.name.endswith(".pt"):
                n += 1
    except FileNotFoundError:
        return 0
    return n


def bar(done: int, total: int, width: int = 40) -> str:
    frac = done / total if total > 0 else 0.0
    filled = int(width * frac)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def fmt_eta(seconds: float) -> str:
    if seconds < 0 or seconds == float("inf"):
        return "?"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}h{m:02d}m{s:02d}s"


def main():
    ap = argparse.ArgumentParser(description="VAE latent caching progress bar.")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="Cache directory")
    ap.add_argument("--total", type=int, default=TOTAL, help="Total samples (default 277713)")
    ap.add_argument("--watch", nargs="?", const=10, type=int, default=None,
                    help="Refresh every N seconds (default 10 if --watch given without value)")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    total = args.total
    prev_done = None
    prev_t = None

    while True:
        done = count_done(cache_dir)
        now = time.time()
        rate = None
        eta = -1
        if prev_done is not None and prev_t is not None and now > prev_t:
            dt = now - prev_t
            dd = done - prev_done
            if dt > 0:
                rate = dd / dt
                if rate > 0:
                    eta = (total - done) / rate

        pct = 100.0 * done / total if total > 0 else 0.0
        rate_str = f"{rate:.1f} samples/s" if rate is not None else "--"
        eta_str = fmt_eta(eta) if rate is not None else "?"
        # \r to overwrite the line in watch mode
        line = (f"\r{bar(done, total)} {done}/{total} ({pct:5.1f}%) "
                f"rate={rate_str} ETA={eta_str}")
        print(line, end="", flush=True)

        if args.watch is None:
            print()  # newline for one-shot mode
            break
        prev_done = done
        prev_t = now
        time.sleep(args.watch)

    # Final newline
    print()


if __name__ == "__main__":
    main()
