import os
import pandas as pd
import wandb

# ── W&B connection ──────────────────────────────────────────────────
api = wandb.Api()
entity  = "fredella-pang-university-of-calgary-in-alberta"
project = "DST Continual Learning"
runs = api.runs(f"{entity}/{project}")

# ── Output root ─────────────────────────────────────────────────────
CSV_ROOT = "csv_export"

# Tasks / sparsifiers we care about
TASKS       = {"CIFAR10", "CIFAR100"}
SPARSIFIERS = {"dense", "static", "gmp", "rigl", "set"}

# ── Pre-create the folder tree ──────────────────────────────────────
for task in TASKS:
    for sparsifier in SPARSIFIERS:
        os.makedirs(os.path.join(CSV_ROOT, task, sparsifier), exist_ok=True)

print(f"Directory tree created under '{CSV_ROOT}/'")

# ── Helpers ─────────────────────────────────────────────────────────
def _cfg(run, key, default=None):
    """Safely get a config value (handles missing keys)."""
    return run.config.get(key, default)


def build_filename(run):
    """
    Build a descriptive CSV filename based on the sparsifier type.

    static → seed, sparsity
    gmp    → seed, sparsity, num_mask_updates
    rigl   → seed, sparsity, pruning_ratio, num_mask_updates
    set    → seed, sparsity, pruning_ratio, num_mask_updates
    """
    sparsifier = _cfg(run, "sparsifier", "unknown")
    seed       = _cfg(run, "seed", 0)
    sparsity   = _cfg(run, "sparsity", 0)

    base = f"seed{seed}_sp{sparsity}"

    if sparsifier == "static":
        return f"{sparsifier}_{base}.csv"

    elif sparsifier == "dense":
        return f"{sparsifier}_seed{seed}.csv"

    elif sparsifier == "gmp":
        nmu = _cfg(run, "num_mask_updates", 0)
        return f"{sparsifier}_{base}_nmu{nmu}.csv"

    elif sparsifier in ("rigl", "set"):
        pr  = _cfg(run, "pruning_ratio", 0)
        nmu = _cfg(run, "num_mask_updates", 0)
        return f"{sparsifier}_{base}_pr{pr}_nmu{nmu}.csv"

    else:
        # Fallback for any unexpected sparsifier
        return f"{sparsifier}_{base}_{run.id}.csv"


# ── Main export loop ────────────────────────────────────────────────
exported = 0
skipped  = 0

for run in runs:
    task       = _cfg(run, "task")
    sparsifier = _cfg(run, "sparsifier")

    # Skip runs that don't match the tasks / sparsifiers we want
    if task not in TASKS or sparsifier not in SPARSIFIERS:
        skipped += 1
        continue

    # Build path early so we can skip already-exported files
    filename = build_filename(run)
    outpath  = os.path.join(CSV_ROOT, task, sparsifier, filename)

    if os.path.exists(outpath):
        skipped += 1
        print(f"  ⏭  {outpath}  (already exists)")
        continue

    # Pull history
    history = run.history(samples=10_000)

    if "test/acc" not in history.columns:
        print(f"  ⚠  Skipping '{run.name}' — no 'test/acc' column")
        skipped += 1
        continue

    df = pd.DataFrame({"accuracy": history["test/acc"]})
    df.to_csv(outpath, index=False)

    exported += 1
    print(f"  ✓  {outpath}")

print(f"\nDone — exported {exported} runs, skipped {skipped}.")