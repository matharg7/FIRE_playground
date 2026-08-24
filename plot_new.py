
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_ROOT   = os.path.join(BASE_DIR, "csv_export")
PLOT_DIR   = os.path.join(BASE_DIR, "plotter", "plots", "svg")

os.makedirs(PLOT_DIR, exist_ok=True)

# ── Seeds (must match what export_data.py wrote) ────────────────────
SEEDS = [0, 5, 8]

# ── Plot styling ────────────────────────────────────────────────────
COLORS = {
    "rigl":   "#D55E00",  # vermilion
    "set":    "#CC79A7",  # reddish purple
    "gmp":    "#009E73",  # bluish green
    "static": "#0072B2",  # blue
}
LABELS = {
    "rigl":   "RigL",
    "set":    "SET",
    "gmp":    "GMP",
    "static": "Static",
}

BLK_LINE_WIDTH   = 1.8
COLOR_LINE_WIDTH = 1.25
DISPLAY_SMOOTHING = 9
X_LIM = (0, 1000)
Y_LIM = (0.1, 0.35)
# Y_LIM = (0.25,.35)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "axes.linewidth": 0.7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


# ── File-path builders (match export_data.py naming) ────────────────
def _dense_path(task, seed):
    return os.path.join(CSV_ROOT, task, "dense", f"dense_seed{seed}.csv")


def _static_path(task, seed, sparsity):
    return os.path.join(CSV_ROOT, task, "static",
                        f"static_seed{seed}_sp{sparsity}.csv")


def _gmp_path(task, seed, sparsity, nmu):
    return os.path.join(CSV_ROOT, task, "gmp",
                        f"gmp_seed{seed}_sp{sparsity}_nmu{nmu}.csv")


def _dst_path(task, sparsifier, seed, sparsity, pruning_ratio, nmu):
    """Path for rigl / set."""
    return os.path.join(CSV_ROOT, task, sparsifier,
                        f"{sparsifier}_seed{seed}_sp{sparsity}"
                        f"_pr{pruning_ratio}_nmu{nmu}.csv")


# ── Generic loader: aggregate across seeds ──────────────────────────
def load_and_aggregate(path_fn):
    """
    Args:
        path_fn: callable(seed) -> filepath

    Returns:
        (mean_array, std_array) or (None, None)
    """
    dfs = []
    for seed in SEEDS:
        fp = path_fn(seed)
        if os.path.isfile(fp):
            df = pd.read_csv(fp)
            dfs.append(df["accuracy"])
        else:
            pass  # silently skip missing seeds

    if not dfs:
        return None, None

    min_len = min(len(d) for d in dfs)
    dfs = [d[:min_len] for d in dfs]
    stacked = np.stack(dfs, axis=0)
    return np.mean(stacked, axis=0), np.std(stacked, axis=0)


    def _plot_series(ax, mean, std, label, color, linewidth=COLOR_LINE_WIDTH,
             zorder=4):
        """Draw a legible mean curve and a restrained across-seed band."""
        x = np.arange(len(mean))
        mean = pd.Series(mean).rolling(DISPLAY_SMOOTHING, center=True,
                       min_periods=1).mean().to_numpy()
        std = pd.Series(std).rolling(DISPLAY_SMOOTHING, center=True,
                     min_periods=1).mean().to_numpy()
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.14,
                linewidth=0, zorder=zorder - 2)
        ax.plot(x, mean, label=label, color=color, linewidth=linewidth,
            solid_capstyle="round", zorder=zorder)


# ── Subplot renderer ────────────────────────────────────────────────
def plot_subplot(ax, task, sparsity, nmu, pruning_ratio, dense_mean, dense_std):
    """Plot dense baseline + all sparsifiers on one axis."""

    # Dense baseline
    if dense_mean is not None:
        _plot_series(ax, dense_mean, dense_std, "Dense", "#222222",
                 linewidth=BLK_LINE_WIDTH, zorder=10)

    # Static (only varies by sparsity, not nmu)
    mean, std = load_and_aggregate(
        lambda s: _static_path(task, s, sparsity))
    if mean is not None:
        _plot_series(ax, mean, std, LABELS["static"], COLORS["static"])

    # GMP (varies by sparsity + nmu)
    mean, std = load_and_aggregate(
        lambda s: _gmp_path(task, s, sparsity, nmu))
    if mean is not None:
        _plot_series(ax, mean, std, LABELS["gmp"], COLORS["gmp"])

    # RigL / SET (vary by sparsity + pruning_ratio + nmu)
    for name in ("rigl", "set"):
        mean, std = load_and_aggregate(
            lambda s, _n=name: _dst_path(task, _n, s, sparsity, pruning_ratio, nmu))
        if mean is not None:
            _plot_series(ax, mean, std, LABELS[name], COLORS[name])

    ax.set_xlim(X_LIM)
    ax.set_ylim(Y_LIM)
    ax.grid(which="major", color="#7f8790", alpha=0.28, linewidth=0.45)
    ax.grid(which="minor", color="#aeb4ba", alpha=0.18, linewidth=0.3)
    ax.minorticks_on()
    ax.set_axisbelow(True)


# ── Grid plot (NMU rows × Sparsity cols) ────────────────────────────
def main_grid(task="CIFAR100", pruning_ratio=0.3,
              sparsity_list=None, nmu_list=None):
    """
    Create a grid of subplots for one task.
    Rows = nmu values, Columns = sparsity values.
    """
    if sparsity_list is None:
        sparsity_list = [0.1,0.3,0.5, 0.7, 0.9,0.95, 0.99]
    if nmu_list is None:
        nmu_list = [10, 100, 1000, 10000, 100000]

    n_rows = len(nmu_list)
    n_cols = len(sparsity_list)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.35, n_rows * 1.95),
                             sharex=True, sharey=True)

    # Ensure axes is 2D even for 1-row or 1-col grids
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    # Dense baseline (independent of sparsity / nmu)
    dense_mean, dense_std = load_and_aggregate(
        lambda s: _dense_path(task, s))

    for row, nmu_val in enumerate(nmu_list):
        for col, sp_val in enumerate(sparsity_list):
            ax = axes[row, col]
            plot_subplot(ax, task, sp_val, nmu_val, pruning_ratio,
                         dense_mean, dense_std)

            if row == 0:
                ax.set_title(f"Sparsity = {sp_val:g}", fontweight="bold",
                             pad=5)
            if col == 0:
                ax.annotate(f"NMU = {nmu_val:g}", xy=(0, 0.5),
                            xytext=(-34, 0), xycoords="axes fraction",
                            textcoords="offset points", ha="right",
                            va="center", rotation=90, fontsize=7)

    fig.supxlabel("Training steps", fontsize=9, y=0.015)
    fig.supylabel("Test accuracy", fontsize=9, x=0.012)

    # Use a single compact legend for the entire figure.
    handles, leg_labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, leg_labels, loc="upper center",
                   ncol=len(leg_labels), frameon=False,
                   bbox_to_anchor=(0.5, 0.985), handlelength=2.2,
                   columnspacing=1.2)

    fig.suptitle(
        f"{task}  |  Pruning Ratio: {pruning_ratio}  |  "
        f"Rows: NMU  |  Columns: Sparsity",
        fontsize=12, fontweight="bold", y=0.995,
    )

    fig.tight_layout(rect=(0.045, 0.045, 0.995, 0.94), w_pad=0.8,
                     h_pad=0.8)
    output_stem = os.path.join(PLOT_DIR,
                               f"{task}_grid_pr{pruning_ratio}")
    fig.savefig(f"{output_stem}.svg", bbox_inches="tight")
    fig.savefig(f"{output_stem}.pdf", bbox_inches="tight")
    plt.close()

    
    # plt.tight_layout()
    # # Uncomment to save:
    # plt.savefig(os.path.join(PLOT_DIR,
    #     f"{task}_grid_pr{pruning_ratio}_zoomed.png"), bbox_inches="tight", dpi=150)
    # plt.show()


# ── Single-plot mode ────────────────────────────────────────────────
def main(task="CIFAR10", sparsity=0.9, pruning_ratio=0.3, nmu=10000,
         zoom=False):
    """Single-panel plot for one (task, sparsity, pruning_ratio, nmu) combo."""
    if zoom:
        xlim, ylim, zoom_text = (700, 1000), (0.7, 0.9), "_Zoomed"
    else:
        xlim, ylim, zoom_text = X_LIM, Y_LIM, ""

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)

    # Dense baseline
    dense_mean, dense_std = load_and_aggregate(
        lambda s: _dense_path(task, s))
    if dense_mean is not None:
        _plot_series(ax, dense_mean, dense_std, "Dense", "#222222",
                     linewidth=BLK_LINE_WIDTH, zorder=10)
    # Static
    mean, std = load_and_aggregate(
        lambda s: _static_path(task, s, sparsity))
    if mean is not None:
        _plot_series(ax, mean, std, LABELS["static"], COLORS["static"])
    # GMP
    mean, std = load_and_aggregate(
        lambda s: _gmp_path(task, s, sparsity, nmu))
    if mean is not None:
        _plot_series(ax, mean, std, LABELS["gmp"], COLORS["gmp"])
    # RigL / SET
    for name in ("rigl", "set"):
        mean, std = load_and_aggregate(
            lambda s, _n=name: _dst_path(task, _n, s, sparsity, pruning_ratio, nmu))
        if mean is not None:
            _plot_series(ax, mean, std, LABELS[name], COLORS[name])
    ax.legend(frameon=False, ncol=5, loc="lower right")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(which="major", color="#7f8790", alpha=0.28, linewidth=0.5)
    ax.grid(which="minor", color="#aeb4ba", alpha=0.18, linewidth=0.3)
    ax.minorticks_on()
    ax.set_axisbelow(True)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"{task}  |  Sparsity = {sparsity:g}  "
              f"Pruning Ratio: {pruning_ratio}  NMU: {nmu}")
    # Uncomment to save:
    # plt.savefig(os.path.join(PLOT_DIR,
    #     f"{task}_s{sparsity}_pr{pruning_ratio}_nmu{nmu}{zoom_text}.png"))
    plt.show()


if __name__ == "__main__":
    # --- Grid for each task ---
    for t in ("CIFAR100", ):#"CIFAR100"
        for r in [0.1, 0.3, 0.5, 0.7, 0.9]:
            main_grid(task=t, pruning_ratio=r)

    # --- Or single plot ---
    # main(task="CIFAR10", sparsity=0.9, pruning_ratio=0.3, nmu=10000)
