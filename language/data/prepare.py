"""Tokenize the training corpora into flat uint16 .bin files.

    python prepare.py --download-only              # warm the HF cache (needs internet)
    python prepare.py --out-dir /path/to/data      # tokenize into <out-dir>/<name>/

Splitting download from tokenization matters on clusters whose compute nodes
have no internet: download on a login node, tokenize in a CPU job.

Each output is a flat array of GPT-2 BPE token ids, uint16 because the vocab
(50257) fits, with documents separated by <|endoftext|>. train.py and
train_sparse.py read them with np.memmap, so nothing is loaded into RAM.
"""
import argparse
import os

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

import tiktoken

# huggingface_hub >= 1.x requires namespaced repo ids; the bare "wikitext" and
# "openwebtext" aliases raise HfUriError.
DATASETS = {
    "wikitext": dict(repo="Salesforce/wikitext", config="wikitext-103-v1"),
    "openwebtext": dict(repo="Skylion007/openwebtext", config=None),
}


def build(name, out_dir, num_proc, download_only):
    spec = DATASETS[name]
    print(f"==> {name}: {spec['repo']}" + (f" ({spec['config']})" if spec["config"] else ""))
    dataset = load_dataset(spec["repo"], spec["config"], num_proc=num_proc)
    if download_only:
        print(f"    cached: {sum(len(s) for s in dataset.values()):,} rows")
        return

    enc = tiktoken.get_encoding("gpt2")
    # Both corpora ship only a train split, so carve a small val set out of it.
    split = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split["val"] = split.pop("test")

    def tokenize(example):
        ids = enc.encode_ordinary(example["text"])
        ids.append(enc.eot_token)
        return {"ids": ids, "len": len(ids)}

    tokenized = split.map(tokenize, remove_columns=["text"],
                          desc=f"tokenizing {name}", num_proc=num_proc)

    target = os.path.join(out_dir, name)
    os.makedirs(target, exist_ok=True)
    for split_name, dset in tokenized.items():
        total = int(np.sum(dset["len"], dtype=np.uint64))
        path = os.path.join(target, f"{split_name}.bin")
        arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(total,))
        idx, batches = 0, 1024
        for batch_idx in tqdm(range(batches), desc=f"writing {path}"):
            batch = dset.shard(num_shards=batches, index=batch_idx,
                               contiguous=True).with_format("numpy")
            chunk = np.concatenate(batch["ids"])
            arr[idx:idx + len(chunk)] = chunk
            idx += len(chunk)
        arr.flush()
        print(f"    {path}: {total:,} tokens ({total * 2 / 1e9:.2f} GB)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    p.add_argument("--out-dir", default=os.environ.get("FIRE_DATA_DIR", "data"))
    p.add_argument("--num-proc", type=int,
                   default=min(8, len(os.sched_getaffinity(0))))
    p.add_argument("--download-only", action="store_true",
                   help="populate the HF cache and stop (run this where there is internet)")
    args = p.parse_args()

    for name in args.datasets:
        build(name, args.out_dir, args.num_proc, args.download_only)


if __name__ == "__main__":
    main()
