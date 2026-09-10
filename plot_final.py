"""
Publication-quality figures for the DST continual-learning experiments.

Three figures, built from the CSVs exported by export_data.py:

  1. hero_cifar100_s05_nmu1000  — single-panel CIFAR100 run at sparsity 0.5,
     pruning_ratio 0.9, NMU 1000: RigL/SET pull ahead of Dense while the
     topology-frozen Static/GMP baselines do not. Includes a zoomed inset
     on the final stages to make the separation legible.
  2. accuracy_grid_cifar100      — 3x3 grid (rows = NMU, cols = sparsity) of
     test accuracy vs. epoch, Dense + all four sparse-training methods.
  3. itop_grid_cifar100          — same 3x3 grid, ITOP rate vs. epoch.
     Requires csv_export/itop/CIFAR100/... (produced by export_data.py).

Color palette validated with the dataviz skill's CVD/contrast checker
(scripts/validate_palette.py) against the white chart surface #ffffff:
blue/orange/aqua/violet passes the CVD and normal-vision floors (adjacent
pairs, as appropriate for overlapping line series).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
CSV_ROOT      = os.path.join(BASE_DIR, "csv_export")
ITOP_CSV_ROOT = os.path.join(BASE_DIR, "csv_export", "itop")
PLOT_DIR      = os.path.join(BASE_DIR, "plotter", "plots", "final")
os.makedirs(PLOT_DIR, exist_ok=True)

SEEDS = [0, 5, 8]

# ── Validated categorical palette (dataviz skill) ───────────────────
# Dense stays neutral ink — it's the reference baseline, not a competing
# series. The four sparse-training methods take distinct, CVD-safe hues
# (blue/orange/aqua/violet) validated against #ffffff: worst adjacent CVD
# ΔE 9.2, worst normal-vision ΔE 27.6 (both clear the >=8 / >=15 gates).
# Aqua sits below 3:1 contrast on white (WARN) — mitigated, as before, by
# the always-present legend + thick lines (the relief rule).
INK           = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK     = "#898781"
GRIDLINE      = "#e1e0d9"
BASELINE      = "#c3c2b7"
SURFACE       = "#ffffff"

COLORS = {
    "dense":  INK,
    "static": "#2a78d6",
    "gmp":    "#eb6834",
    "rigl":   "#1baf7a",
    "set":    "#4a3aa7",
}
LABELS = {
    "dense":  "Dense",
    "static": "Static",
    "gmp":    "GMP",
    "rigl":   "RigL",
    "set":    "SET",
}
ORDER = ["dense", "static", "gmp", "rigl", "set"]

LINE_KW = dict(solid_capstyle="round", solid_joinstyle="round")
FILL_ALPHA = 0.12

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor":   SURFACE,
    "axes.edgecolor":   BASELINE,
    "axes.labelcolor":  SECONDARY_INK,
    "text.color":       INK,
    "xtick.color":      MUTED_INK,
    "ytick.color":      MUTED_INK,
    "font.family":      "sans-serif",
    "axes.titleweight": "bold",
    "svg.fonttype":     "none",
})


# ── File-path builders (match export_data.py naming) ────────────────
def _dense_path(root, task, seed):
    return os.path.join(root, task, "dense", f"dense_seed{seed}.csv")


def _static_path(root, task, seed, sparsity):
    return os.path.join(root, task, "static", f"static_seed{seed}_sp{sparsity}.csv")


def _gmp_path(root, task, seed, sparsity, nmu):
    return os.path.join(root, task, "gmp", f"gmp_seed{seed}_sp{sparsity}_nmu{nmu}.csv")


def _dst_path(root, task, sparsifier, seed, sparsity, pruning_ratio, nmu):
    return os.path.join(root, task, sparsifier,
                         f"{sparsifier}_seed{seed}_sp{sparsity}"
                         f"_pr{pruning_ratio}_nmu{nmu}.csv")


def load_and_aggregate(path_fn):
    dfs = []
    for seed in SEEDS:
        fp = path_fn(seed)
        if os.path.isfile(fp) and os.path.getsize(fp) > 0:
            dfs.append(pd.read_csv(fp)["accuracy"])
    if not dfs:
        return None, None
    min_len = min(len(d) for d in dfs)
    stacked = np.stack([d[:min_len] for d in dfs], axis=0)
    return np.mean(stacked, axis=0), np.std(stacked, axis=0)


def load_all_series(root, task, sparsity, nmu, pruning_ratio):
    """Return {method: (mean, std)} for dense + the four sparse methods."""
    series = {}
    series["dense"] = load_and_aggregate(lambda s: _dense_path(root, task, s))
    series["static"] = load_and_aggregate(lambda s: _static_path(root, task, s, sparsity))
    series["gmp"] = load_and_aggregate(lambda s: _gmp_path(root, task, s, sparsity, nmu))
    for name in ("rigl", "set"):
        series[name] = load_and_aggregate(
            lambda s, _n=name: _dst_path(root, task, _n, s, sparsity, pruning_ratio, nmu))
    return series


def _draw_series(ax, series, order=ORDER, linewidth=2.0, dense_linewidth=2.4, zorder0=5):
    for i, name in enumerate(order):
        mean, std = series.get(name, (None, None))
        if mean is None:
            continue
        x = np.arange(len(mean))
        lw = dense_linewidth if name == "dense" else linewidth
        z = zorder0 + (10 if name == "dense" else i)
        ax.plot(x, mean, color=COLORS[name], linewidth=lw, alpha=0.95, zorder=z, **LINE_KW)
        ax.fill_between(x, mean - std, mean + std, color=COLORS[name],
                         alpha=FILL_ALPHA, zorder=1, linewidth=0)


def _style_axes(ax, xlim=None, ylim=None):
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True, color=GRIDLINE, linestyle="-", linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(length=0)


def _legend_handles(order=ORDER):
    return [Line2D([0], [0], color=COLORS[n], linewidth=2.4, **LINE_KW) for n in order], \
           [LABELS[n] for n in order]


# ═══════════════════════════════════════════════════════════════════
# Figure 1 — hero single-panel plot
# ═══════════════════════════════════════════════════════════════════
def plot_hero(task="CIFAR100", sparsity=0.5, pruning_ratio=0.9, nmu=1000, zoom_inset=None):
    series = load_all_series(CSV_ROOT, task, sparsity, nmu, pruning_ratio)
    if series["dense"][0] is None:
        print(f"  ⚠  hero plot: missing dense data for {task}")
        return

    fig, ax = plt.subplots(figsize=(11, 6.5))

    _draw_series(ax, series, linewidth=2.2, dense_linewidth=2.6)

    all_means = [m for m, _ in series.values() if m is not None]
    ymin = min(m.min() for m in all_means)
    ymax = max(m.max() for m in all_means)
    pad = (ymax - ymin) * 0.08
    full_ylim = (max(0, ymin - pad), ymax + pad)
    _style_axes(ax, xlim=(0, 1000), ylim=full_ylim)

    ax.set_xlabel("Epoch", fontsize=19)
    ax.set_ylabel("Test accuracy", fontsize=19)
    ax.tick_params(axis="both", labelsize=16)

    handles, labels = _legend_handles()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=15,
               frameon=False, bbox_to_anchor=(0.5, 0.97))

    # Zoomed inset on the final stages — needed when methods separate by only
    # a fraction of a point (e.g. CIFAR10), which is invisible at full scale.
    if zoom_inset is not None:
        x0, x1 = zoom_inset
        axins = ax.inset_axes([0.50, 0.10, 0.45, 0.42])
        _draw_series(axins, series, linewidth=1.8, dense_linewidth=2.2)
        zoom_vals = [m[x0:x1] for m in all_means if len(m) >= x1]
        zmin = min(v.min() for v in zoom_vals)
        zmax = max(v.max() for v in zoom_vals)
        zpad = (zmax - zmin) * 0.12
        _style_axes(axins, xlim=(x0, x1), ylim=(zmin - zpad, zmax + zpad))
        axins.set_xticklabels([])
        axins.tick_params(axis="y", labelsize=12)
        for spine in axins.spines.values():
            spine.set_visible(True)
            spine.set_color(BASELINE)
        ax.indicate_inset_zoom(axins, edgecolor=SECONDARY_INK, alpha=0.6)

    fig.subplots_adjust(top=0.91, bottom=0.10, left=0.09, right=0.96)
    for ext in ("svg", "png", "pdf"):
        fig.savefig(os.path.join(PLOT_DIR, f"hero_{task}_s{sparsity}_nmu{nmu}.{ext}"),
                    bbox_inches="tight", dpi=220, facecolor=SURFACE)
    plt.close(fig)
    print(f"  ✓ hero plot saved for {task} s={sparsity} nmu={nmu}")


# ═══════════════════════════════════════════════════════════════════
# Figure 2 / 3 — grid plots (rows = NMU, cols = sparsity)
# ═══════════════════════════════════════════════════════════════════
def _plot_grid_generic(out_name, ylabel_metric, row_values, sparsity_list,
                        row_label_fn, series_fn, highlight=None,
                        y_from_zero=True, xlim=(0, 1000), share_y="all"):
    n_rows, n_cols = len(row_values), len(sparsity_list)
    sharey_arg = "col" if share_y == "col" else True
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.2, n_rows * 3.2),
                              sharex=True, sharey=sharey_arg)
    axes = np.atleast_2d(axes)

    grid_series = {}
    col_values = {col: [] for col in range(n_cols)}
    for row, row_val in enumerate(row_values):
        for col, sp_val in enumerate(sparsity_list):
            s = series_fn(row_val, sp_val)
            grid_series[(row, col)] = s
            for m, _ in s.values():
                if m is not None:
                    x0i, x1i = max(0, xlim[0]), min(len(m), xlim[1])
                    if x1i > x0i:
                        col_values[col].append(m[x0i:x1i])

    if not any(col_values.values()):
        print(f"  ⚠  grid plot '{out_name}' skipped — no data found")
        plt.close(fig)
        return

    def _ylim_for(values):
        cat = np.concatenate(values)
        # Robust range: 0.5th/99.5th percentile so a single noisy transient
        # (e.g. one seed's post-restart dip) can't blow out the whole scale.
        lo, hi = np.percentile(cat, [0.5, 99.5])
        ymin = 0.0 if y_from_zero else lo
        pad = (hi - ymin) * 0.06
        top = min(1.0, hi + pad) if hi <= 1 else hi + pad
        return (max(0, ymin - (0 if y_from_zero else pad)), top)

    if share_y == "col":
        ylim_by_col = {col: _ylim_for(vals) for col, vals in col_values.items() if vals}
    else:
        global_ylim = _ylim_for([v for vals in col_values.values() for v in vals])
        ylim_by_col = {col: global_ylim for col in range(n_cols)}

    for row, row_val in enumerate(row_values):
        for col, sp_val in enumerate(sparsity_list):
            ax = axes[row, col]
            _draw_series(ax, grid_series[(row, col)], linewidth=1.6, dense_linewidth=2.0)
            _style_axes(ax, xlim=xlim, ylim=ylim_by_col[col])
            ax.tick_params(axis="both", labelsize=12)

            if highlight == (row_val, sp_val):
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color(COLORS["rigl"])
                    spine.set_linewidth(1.8)
                ax.set_facecolor("#fff6f4")

            if row == 0:
                ax.set_title(f"Sparsity {sp_val}", fontsize=15, fontweight="bold",
                             color=SECONDARY_INK, pad=10)
            if col == 0:
                ax.set_ylabel(row_label_fn(row_val), fontsize=14, fontweight="bold",
                              color=SECONDARY_INK)

    fig.supylabel(ylabel_metric, fontsize=15, color=SECONDARY_INK, x=0.005)
    fig.supxlabel("Epochs", fontsize=15, color=SECONDARY_INK)

    handles, labels = _legend_handles()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=15,
               frameon=False, bbox_to_anchor=(0.5, 0.99))

    fig.tight_layout(rect=[0.01, 0, 1, 0.94])
    for ext in ("svg", "png", "pdf"):
        fig.savefig(os.path.join(PLOT_DIR, f"{out_name}.{ext}"),
                    bbox_inches="tight", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"  ✓ grid plot saved: {out_name}")


def plot_grid(root, ylabel_metric, out_name, task="CIFAR100",
              pruning_ratio=0.9, sparsity_list=(0.1, 0.5, 0.95),
              nmu_list=(10, 1000, 10000), highlight=None, y_from_zero=True,
              xlim=(0, 1000), share_y="all"):
    """Grid with rows = NMU, cols = sparsity, at a fixed pruning_ratio (drop fraction)."""
    _plot_grid_generic(
        out_name=out_name, ylabel_metric=ylabel_metric,
        row_values=nmu_list, sparsity_list=sparsity_list,
        row_label_fn=lambda nmu_val: f"NMU {nmu_val}",
        series_fn=lambda nmu_val, sp_val: load_all_series(root, task, sp_val, nmu_val, pruning_ratio),
        highlight=highlight, y_from_zero=y_from_zero, xlim=xlim, share_y=share_y,
    )


def plot_grid_drop_fraction(root, ylabel_metric, out_name, task="CIFAR100",
                             nmu=1000, sparsity_list=(0.1, 0.5, 0.95),
                             pruning_ratio_list=(0.1, 0.3, 0.5, 0.7, 0.9),
                             highlight=None, y_from_zero=True,
                             xlim=(0, 1000), share_y="all"):
    """Grid with rows = drop fraction (pruning_ratio), cols = sparsity, at a fixed NMU.

    Static/GMP/Dense don't depend on pruning_ratio, so they repeat identically
    down each column — only the RigL/SET curves change row to row.
    """
    _plot_grid_generic(
        out_name=out_name, ylabel_metric=ylabel_metric,
        row_values=pruning_ratio_list, sparsity_list=sparsity_list,
        row_label_fn=lambda pr_val: f"Drop frac {pr_val}",
        series_fn=lambda pr_val, sp_val: load_all_series(root, task, sp_val, nmu, pr_val),
        highlight=highlight, y_from_zero=y_from_zero, xlim=xlim, share_y=share_y,
    )


if __name__ == "__main__":
    plot_hero(task="CIFAR100", sparsity=0.5, pruning_ratio=0.9, nmu=1000)

    # CIFAR10 hero: sp=0.1, pr=0.3, nmu=100 — RigL pulls ahead of Dense while
    # Static/GMP lag behind (pr=0.3 gives a more consistent RigL/SET margin
    # across the NMU sweep than pr=0.9 — see accuracy_grid_full_CIFAR10).
    # Margins are much smaller than CIFAR100's since dense CIFAR10 already
    # saturates near 82%, so a zoomed inset on the final epochs makes the
    # gap legible.
    plot_hero(task="CIFAR10", sparsity=0.1, pruning_ratio=0.3, nmu=100,
              zoom_inset=(700, 1000))

    plot_grid(
        root=CSV_ROOT,
        ylabel_metric="Test accuracy",
        out_name="accuracy_grid_CIFAR100",
        task="CIFAR100",
        pruning_ratio=0.9,
        sparsity_list=(0.1, 0.5, 0.95),
        nmu_list=(10, 1000, 10000),
        highlight=(1000, 0.5),
        y_from_zero=True,
    )

    plot_grid(
        root=CSV_ROOT,
        ylabel_metric="Test accuracy",
        out_name="accuracy_grid_zoom_CIFAR10",
        task="CIFAR10",
        pruning_ratio=0.9,
        sparsity_list=(0.1, 0.7, 0.95),
        nmu_list=(10, 100, 1000),
        highlight=None,
        y_from_zero=False,
        xlim=(700, 1000),
        share_y="col",
    )

    # Full-sweep grids — every sparsity and NMU value swept for each
    # benchmark. CIFAR100 uses pruning_ratio=0.9 (best accuracy, and RigL/SET
    # were swept at every NMU there); CIFAR10 uses 0.3 instead (see
    # CIFAR10_FULL_PR below) since RigL/SET were only swept up to NMU=10000
    # at pruning_ratio<=0.5.
    FULL_SPARSITY = (0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99)

    plot_grid(
        root=CSV_ROOT,
        ylabel_metric="Test accuracy",
        out_name="accuracy_grid_full_CIFAR100",
        task="CIFAR100",
        pruning_ratio=0.9,
        sparsity_list=FULL_SPARSITY,
        nmu_list=(10, 100, 1000, 10000, 100000),
        highlight=(1000, 0.5),
        y_from_zero=True,
    )

    # CIFAR10 RigL/SET were only swept up to NMU=10000, and only at
    # pruning_ratio<=0.5 — pr=0.9 (used elsewhere) caps out at NMU=1000 for
    # those two methods. pr=0.3 is the ratio with full NMU coverage, so the
    # full-sweep CIFAR10 grids use it instead of 0.9.
    CIFAR10_FULL_PR = 0.3

    plot_grid(
        root=CSV_ROOT,
        ylabel_metric="Test accuracy",
        out_name="accuracy_grid_full_CIFAR10",
        task="CIFAR10",
        pruning_ratio=CIFAR10_FULL_PR,
        sparsity_list=FULL_SPARSITY,
        nmu_list=(10, 100, 1000, 10000),
        highlight=None,
        y_from_zero=False,
        xlim=(700, 1000),
        share_y="col",
    )

    plot_grid(
        root=ITOP_CSV_ROOT,
        ylabel_metric="ITOP rate",
        out_name="itop_grid_full_CIFAR100",
        task="CIFAR100",
        pruning_ratio=0.9,
        sparsity_list=FULL_SPARSITY,
        nmu_list=(10, 100, 1000, 10000, 100000),
        highlight=(1000, 0.5),
        y_from_zero=True,
    )

    plot_grid(
        root=ITOP_CSV_ROOT,
        ylabel_metric="ITOP rate",
        out_name="itop_grid_full_CIFAR10",
        task="CIFAR10",
        pruning_ratio=CIFAR10_FULL_PR,
        sparsity_list=FULL_SPARSITY,
        nmu_list=(10, 100, 1000, 10000),
        highlight=(100, 0.1),
        y_from_zero=True,
    )

    # Drop-fraction grids — rows = pruning_ratio (drop fraction), cols =
    # sparsity, with NMU fixed at each benchmark's hero value (the NMU that
    # gives the best accuracy — see plot_hero calls above: CIFAR100 nmu=1000,
    # CIFAR10 nmu=100).
    FULL_PRUNING_RATIO = (0.1, 0.3, 0.5, 0.7, 0.9)

    plot_grid_drop_fraction(
        root=CSV_ROOT,
        ylabel_metric="Test accuracy",
        out_name="accuracy_grid_dropfrac_CIFAR100",
        task="CIFAR100",
        nmu=1000,
        sparsity_list=FULL_SPARSITY,
        pruning_ratio_list=FULL_PRUNING_RATIO,
        highlight=(0.9, 0.5),
        y_from_zero=True,
    )

    plot_grid_drop_fraction(
        root=CSV_ROOT,
        ylabel_metric="Test accuracy",
        out_name="accuracy_grid_dropfrac_zoom_CIFAR10",
        task="CIFAR10",
        nmu=100,
        sparsity_list=FULL_SPARSITY,
        pruning_ratio_list=FULL_PRUNING_RATIO,
        highlight=(0.9, 0.1),
        y_from_zero=False,
        xlim=(700, 1000),
        share_y="col",
    )
