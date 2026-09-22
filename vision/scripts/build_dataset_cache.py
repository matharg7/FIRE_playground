"""Pre-stage the vision datasets ahead of any training job.

Both phases happen lazily on first use, but doing them here instead means the
one-time cost does not burn GPU time inside a training allocation, and a job
array launched against a cold cache cannot have every member pay it at once.

The two phases need different machines, so they are separate invocations:

    # 1. Raw download -- needs internet, which compute nodes may not have.
    python vision/scripts/build_dataset_cache.py --download

    # 2. Decode TinyImageNet to a tensor cache -- needs CPUs, not a GPU.
    salloc --account=<account> --time=0:30:00 --cpus-per-task=8 --mem=16G
    python vision/scripts/build_dataset_cache.py

Phase 2 is a no-op for CIFAR, which torchvision already hands over as one
in-memory array; only TinyImageNet's 110k JPEGs need decoding. Both phases
skip work that is already done, so either is safe to re-run.

Reads and writes $FIRE_DATA_DIR, else $SCRATCH/datasets.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from task import DATA_DIR, TASKS, TinyImageNet  # noqa: E402


def download(names):
    print(f"Staging raw data under {DATA_DIR}")
    for name in names:
        TASKS[name].download()


def build_cache():
    cache_path = os.path.join(
        DATA_DIR, f'tiny-imagenet-200-decoded-v{TinyImageNet.CACHE_VERSION}.pt')
    if os.path.exists(cache_path):
        print(f"Cache already present: {cache_path}")
        return

    dataset_dir = os.path.join(DATA_DIR, 'tiny-imagenet-200')
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise SystemExit(
            f"TinyImageNet is not in {DATA_DIR}. Run this script with --download "
            "from a login node first (compute nodes may have no internet access)."
        )

    TinyImageNet._build_cache(train_dir, val_dir, cache_path)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--download', action='store_true',
        help='download the raw datasets instead of building the decoded cache; '
             'needs internet, so run this on a login node')
    parser.add_argument(
        '--datasets', nargs='+', metavar='NAME',
        choices=sorted(TASKS), default=sorted(TASKS),
        help=f"which datasets to download (default: all of {', '.join(sorted(TASKS))})")
    args = parser.parse_args()

    if args.download:
        download(args.datasets)
    else:
        build_cache()


if __name__ == '__main__':
    main()
