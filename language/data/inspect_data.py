"""
See the pre-tokenized datasets exactly the way train.py sees them.

Run from language/ -- no GPU needed, it only reads the .bin files:

    python data/inspect_data.py                                  # both datasets
    python data/inspect_data.py --datasets wikitext --samples 4  # more random windows
    python data/inspect_data.py --ratio 0.02 --replay 1          # explain a chunk config
    python data/inspect_data.py --plot data_stats.png            # also save plots

Sections:
  1. FILE     a .bin file is nothing but a flat array of uint16 token ids
  2. DOCS     documents are glued end to end, separated by <|endoftext|>
  3. WINDOWS  random (x, y) training windows, drawn the same way as get_batch()
  4. CHUNK    how subset_ratio and data_replay_ratio become tokens and iterations
"""
import argparse
import math
import os

import numpy as np
import tiktoken

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
EOT = 50256  # <|endoftext|>, appended after every row by prepare.py

# (subset_ratio, data_replay_ratio) as used for Fig 3 in warm_start_gpt.sh
PAPER_CHUNKS = {"wikitext": (1.0, 400), "openwebtext": (1.0, 2)}


class Style:
    def __init__(self, enabled):
        self.enabled = enabled

    def __call__(self, text, code):
        return f"\033[{code}m{text}\033[0m" if self.enabled else text


def header(s, title):
    print()
    print(s(f"━━ {title} " + "━" * max(0, 76 - len(title)), "1;36"))


def load(name, split):
    path = os.path.join(DATA_DIR, name, f"{split}.bin")
    if not os.path.exists(path):
        return None
    # Same call as train.py. Nothing is read into RAM here; the OS pages bytes in on access.
    return np.memmap(path, dtype=np.uint16, mode="r")


def render(enc, tokens, s):
    """Decode token by token, alternating colours so BPE boundaries are visible."""
    out = []
    for i, tok in enumerate(tokens):
        if tok == EOT:
            out.append(s("⟨EOT⟩", "1;97;41") if s.enabled else "⟨EOT⟩")
            continue
        text = enc.decode([int(tok)]).replace("\n", "↵")
        out.append(s(text, "30;46" if i % 2 == 0 else "30;43") if s.enabled else text)
    return ("" if s.enabled else "|").join(out)


def section_file(name, enc, s):
    header(s, f"1. FILE · {name}")
    for split in ("train", "val"):
        d = load(name, split)
        if d is None:
            print(f"  {split}.bin  missing")
            continue
        size = f"{d.nbytes / 1e9:.2f} GB" if d.nbytes >= 1e9 else f"{d.nbytes / 1e6:.2f} MB"
        print(f"  {split}.bin  {size:>9}  ->  {len(d):>14,} tokens   (uint16, 2 bytes each)")
    d = load(name, "train")
    head = [int(t) for t in d[:16]]
    print(f"\n  First 16 entries of train.bin, as stored:\n    {head}")
    print(f"  The same 16 entries, decoded (one colour per token):\n    {render(enc, head, s)}")


def section_docs(name, d, n_stat, s):
    header(s, f"2. DOCS · {name}")
    span = np.asarray(d[:n_stat])
    eots = np.flatnonzero(span == EOT)
    # tokens strictly between consecutive EOTs = one document's body
    lengths = np.diff(np.concatenate(([-1], eots))) - 1
    if len(lengths) == 0:
        print("  no <|endoftext|> in the sampled span")
        return span, lengths
    empty = np.mean(lengths == 0)
    print(f"  Scanned the first {len(span):,} tokens and found {len(lengths):,} documents.")
    print(f"  Document length (tokens):  median {int(np.median(lengths)):,} · mean {lengths.mean():,.0f}"
          f" · p90 {int(np.percentile(lengths, 90)):,} · max {int(lengths.max()):,}")
    print(f"  Empty documents (two EOTs in a row): {empty:.1%}")
    longer = np.mean(lengths >= 1024)
    print(f"  Documents at least one full block (1024 tokens) long: {longer:.1%}")
    if empty > 0.2:
        print(s("  Note: this many empty documents means the source rows are lines, not whole articles.", "33"))
    return span, lengths


def section_windows(name, d, enc, s, ratio, block, samples, rng):
    header(s, f"3. WINDOWS · {name} · subset_ratio={ratio}")
    end_index = int(len(d) * ratio) - block
    print(f"  get_batch() picks a start offset uniformly in [0, {end_index:,}) and slices:")
    print(f"    x = data[i : i+{block}]      (the input)")
    print(f"    y = data[i+1 : i+1+{block}]  (the same tokens shifted left by one = the targets)")
    print(f"  train.py uses block_size=1024; showing {block} tokens so it fits on screen.")
    for k in range(samples):
        i = int(rng.integers(end_index))
        x = [int(t) for t in d[i:i + block]]
        y = [int(t) for t in d[i + 1:i + 1 + block]]
        print(f"\n  window {k + 1}: offset {i:,}  ({i / len(d):.2%} into the file)")
        print(f"    {render(enc, x, s)}")
        print("    One window is really many next-token predictions at once:")
        for t in range(min(5, block)):
            ctx = enc.decode(x[max(0, t - 5):t + 1]).replace("\n", "↵")
            tgt = "⟨EOT⟩" if y[t] == EOT else enc.decode([y[t]]).replace("\n", "↵")
            print(f"      position {t}:  …{ctx!r:<38} -> predict {tgt!r}")


def section_chunk(name, d, ratio, replay, tokens_per_iter, s):
    header(s, f"4. CHUNK · {name} · subset_ratio={ratio} · data_replay_ratio={replay}")
    n = len(d)
    subset = int(n * ratio)
    chunk_tokens = int(replay * n * ratio)          # train.py:343
    iters = chunk_tokens // tokens_per_iter          # train.py:346
    warmup = min(int(0.1 * iters), 2000)             # train.py:347
    eval_every = min(2000, iters // 20)              # train.py:349
    width = 50
    filled = max(1, round(width * ratio)) if ratio > 0 else 0
    print(f"  train.bin  [{s('█' * filled, '32')}{'·' * (width - filled)}]  {n:,} tokens")
    print(f"  subset     first {ratio:.2%} of the file = {subset:,} tokens (always sliced from the front)")
    print(f"  chunk      {replay} x {subset:,} = {chunk_tokens:,} tokens to train on")
    if tokens_per_iter:
        print(f"  iterations {chunk_tokens:,} / {tokens_per_iter:,} tokens per iter = {iters:,}")
        print(f"             warmup {warmup:,} iters · eval every {eval_every:,} iters")
    # Each window start is uniform, so coverage of any single token is ~Poisson(replay).
    unseen = math.exp(-replay) if replay < 50 else 0.0
    print(f"  coverage   windows are sampled with replacement, so this is not a true epoch:")
    print(f"             about {unseen:.1%} of the subset is never seen at all, "
          f"and on average each token is seen {replay:g} times")


def plot(stats, enc, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(stats), 3, figsize=(17, 4.6 * len(stats)), squeeze=False)
    for row, (name, (span, lengths)) in enumerate(stats.items()):
        counts = np.bincount(span, minlength=enc.n_vocab)

        ax = axes[row][0]
        top = np.argsort(counts)[::-1][:20]
        labels = ["⟨EOT⟩" if t == EOT else repr(enc.decode([int(t)])) for t in top]
        ax.barh(range(20), counts[top] / len(span) * 100, color="#4C78A8")
        ax.set_yticks(range(20), labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("% of all tokens")
        ax.set_title(f"{name}: 20 most frequent tokens")

        ax = axes[row][1]
        doc_len = lengths + 1  # include the EOT so empty documents still appear at x=1
        bins = np.logspace(0, math.log10(max(doc_len.max(), 2)), 60)
        ax.hist(doc_len, bins=bins, color="#F58518")
        ax.axvline(1024, color="k", ls="--", lw=1)
        ax.text(1024, ax.get_ylim()[1] * 0.9, " block_size", fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("document length in tokens (log)")
        ax.set_ylabel("documents")
        ax.set_title(f"{name}: document lengths")

        ax = axes[row][2]
        freq = np.sort(counts[counts > 0])[::-1]
        ax.loglog(np.arange(1, len(freq) + 1), freq, color="#54A24B")
        ax.set_xlabel("token rank (log)")
        ax.set_ylabel("count (log)")
        ax.set_title(f"{name}: rank vs frequency ({(counts > 0).sum():,} distinct tokens)")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\nSaved plots to {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=["wikitext", "openwebtext"])
    p.add_argument("--samples", type=int, default=2, help="random windows to show per dataset")
    p.add_argument("--block-size", type=int, default=48, help="window length to print (train.py uses 1024)")
    p.add_argument("--ratio", type=float, default=None, help="subset_ratio to sample from / explain")
    p.add_argument("--replay", type=float, default=None, help="data_replay_ratio to explain")
    p.add_argument("--tokens-per-iter", type=int, default=491_520,
                   help="grad_accum * world_size * batch_size * block_size (paper: 8*60*1024)")
    p.add_argument("--stat-tokens", type=int, default=20_000_000, help="tokens scanned for statistics")
    p.add_argument("--plot", default=None, help="save token/document plots to this PNG")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-color", action="store_true")
    args = p.parse_args()

    s = Style(enabled=not args.no_color and os.isatty(1))
    rng = np.random.default_rng(args.seed)
    enc = tiktoken.get_encoding("gpt2")
    stats = {}

    for name in args.datasets:
        if name == "wiki_owt":
            print("\nwiki_owt is not a file on disk: train.py:169-173 picks wikitext for ~1.1% of batches "
                  "and openwebtext for the rest. Inspect the two datasets separately.")
            continue
        d = load(name, "train")
        if d is None:
            print(f"\n{name}: data/{name}/train.bin not found, run data/{name}/prepare.py first")
            continue
        default_ratio, default_replay = PAPER_CHUNKS.get(name, (1.0, 1))
        ratio = args.ratio if args.ratio is not None else default_ratio
        replay = args.replay if args.replay is not None else default_replay

        section_file(name, enc, s)
        stats[name] = section_docs(name, d, args.stat_tokens, s)
        section_windows(name, d, enc, s, ratio, args.block_size, args.samples, rng)
        section_chunk(name, d, ratio, replay, args.tokens_per_iter, s)

    if args.plot and stats:
        plot(stats, enc, args.plot)


if __name__ == "__main__":
    main()
