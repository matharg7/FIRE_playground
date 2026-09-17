"""Build the decoded TinyImageNet tensor cache ahead of any training job.

The cache is built lazily on first use, but doing it here instead means the
one-time decode does not burn GPU time inside a training allocation, and a job
array launched against a cold cache cannot have every member decode at once.

Needs no GPU:

    salloc --account=<account> --time=0:30:00 --cpus-per-task=8 --mem=16G
    python vision/scripts/build_dataset_cache.py

Writes to $FIRE_DATA_DIR, else $SCRATCH/datasets.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from task import DATA_DIR, TinyImageNet  # noqa: E402


def main():
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
            f"TinyImageNet is not in {DATA_DIR}. Download it from a login node "
            "first (compute nodes may have no internet access)."
        )

    TinyImageNet._build_cache(train_dir, val_dir, cache_path)


if __name__ == '__main__':
    main()
