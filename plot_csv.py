
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, "csv")

SEEDS = [0, 5, 8]

# Dense path template (seed placeholder)
DENSE_TEMPLATE = os.path.join(CSV_DIR, "dense", "RESNET18_CIFAR10_dense_seed{seed}.csv")

colors = ["red", "purple", "green", "blue"]
labels = ["rigl", "set", "gmp", "static"]

DEF_SPARSITY, DEF_PRUNING_RATIO, NMU, DEF_DT, DEF_GDT = 0, 0, 0, 0, 0
X_LIM = (0, 1000)
Y_LIM = (0, 0.9)

# Clean dictionary-based nmu -> (dt, gdt) mapping
NMU_TO_DT = {
    100000: (1, 1),
    10000:  (8, 6),
    1000:   (86, 64),
    100:    (864, 648),
    10:     (8640, 6480),
}

# --- Original nmu logic (commented out as fallback) ---
# def set(param, value):
#     global DEF_SEED, DEF_SPARSITY, DEF_PRUNING_RATIO, DEF_DT, DEF_GDT
#     if param == "seed":
#         DEF_SEED = value
#     elif param == "sparsity":
#         DEF_SPARSITY = value
#     elif param == "pruning_ratio":
#         DEF_PRUNING_RATIO = value
#     elif param == "nmu":
#         if (value==10000):
#             DEF_DT = 1
#             DEF_GDT=1
#             return (1,1)
#         elif (value ==1000):
#             DEF_DT = 8
#             DEF_GDT= 6
#             return (8,6)
#         elif (value == 100):
#             DEF_DT = 86
#             DEF_GDT= 64
#             return (86,64)
#         elif (value == 10):
#             DEF_DT = 864
#             DEF_GDT= 648
#             return (864,648)
#         else:
#             DEF_DT = 8640
#             DEF_GDT= 6480
#             return (8640,6480)
#     else:
#         print("Invalid parameter")
# ---


def set_param(param, value):
    """Set a single parameter by name."""
    global DEF_SPARSITY, DEF_PRUNING_RATIO, DEF_DT, DEF_GDT, NMU
    if param == "sparsity":
        DEF_SPARSITY = value

    elif param == "pruning_ratio":
        DEF_PRUNING_RATIO = value
    elif param == "nmu":
        if value not in NMU_TO_DT:
            print(f"Warning: nmu={value} not in NMU_TO_DT mapping, defaulting to nmu=1")
            value = 10
        NMU = value
        DEF_DT, DEF_GDT = NMU_TO_DT[value]
    elif param == "delta_t":
        DEF_DT = value
    elif param == "gdt":
        DEF_GDT = value
    else:
        print(f"Invalid parameter: {param}")


def set_params(sparsity, pruning_ratio, nmu):
    """Set all parameters at once."""
    set_param("sparsity", sparsity)
    set_param("pruning_ratio", pruning_ratio)
    set_param("nmu", nmu)


def build_path_template(sparsifier):
    """
    Build a file path template with a {seed} placeholder for the given sparsifier.
    
    Returns the path template string.
    """
    if sparsifier == "rigl" or sparsifier == "set":
        return os.path.join(
            CSV_DIR, sparsifier,
            f"RESNET18_CIFAR10_dst_sparsity_{DEF_SPARSITY}"
            f"_pruning_ratio_{DEF_PRUNING_RATIO}"
            f"_delta_t_{DEF_DT}"
            f"_sparsifier_{sparsifier}"
            f"_seed{{seed}}.csv"
        )
    elif sparsifier == "gmp":
        return os.path.join(
            CSV_DIR, "gmp",
            f"RESNET18_CIFAR10_gmp_accel_sparsity_0.0"
            f"_final_sparsity_{DEF_SPARSITY}"
            f"_delta_t_{DEF_GDT}"
            f"_sparsifier_gmp"
            f"_seed{{seed}}.csv"
        )
    elif sparsifier == "static":
        return os.path.join(
            CSV_DIR, "static",
            f"RESNET18_CIFAR10_static_sparsity_{DEF_SPARSITY}"
            f"_sparsifier_static"
            f"_seed{{seed}}.csv"
        )
    elif sparsifier == "dense":
        return DENSE_TEMPLATE
    else:
        raise ValueError(f"Unknown sparsifier: {sparsifier}")


def load_and_aggregate(path_template):
    """
    Load accuracy data from all seed files matching the template,
    then compute the mean and std across seeds.
    
    Args:
        path_template: File path string with {seed} placeholder.
    
    Returns:
        (mean_series, std_series) or (None, None) if no files found.
    """
    dfs = []
    for seed in SEEDS:
        filepath = path_template.format(seed=seed)
        if os.path.isfile(filepath):
            df = pd.read_csv(filepath)
            dfs.append(df["accuracy"])
        else:
            print(f"[WARNING] Missing file: {filepath}")

    if len(dfs) == 0:
        return None, None

    # Stack all seed runs into a 2D array (seeds x steps)
    stacked = np.stack(dfs, axis=0)
    mean = np.mean(stacked, axis=0)
    std = np.std(stacked, axis=0)

    return mean, std


def plot_subplot(ax, dense_mean, dense_std):
    """Plot all sparsifier lines + dense baseline on a single subplot axis."""

    # --- Dense baseline (mean ± std) ---
    if dense_mean is not None:
        x = np.arange(len(dense_mean))
        ax.plot(x, dense_mean, label="Dense", color='black', linestyle='-', linewidth=0.4, alpha=0.7, zorder=10)
        ax.fill_between(x, dense_mean - dense_std, dense_mean + dense_std, color='black', alpha=0.2, zorder=0)

    # --- Sparsifiers (mean ± std) ---
    sparsifier_names = ["rigl", "set", "gmp", "static"]
    for i, name in enumerate(sparsifier_names):
        template = build_path_template(name)
        mean, std = load_and_aggregate(template)
        if mean is not None:
            x = np.arange(len(mean))
            ax.plot(x, mean, label=labels[i], color=colors[i], linestyle='-', linewidth=0.3, alpha=0.7, zorder=8)
            ax.fill_between(x, mean - std, mean + std, color=colors[i], alpha=0.2, zorder=2)

    ax.set_xlim(X_LIM)
    ax.set_ylim(Y_LIM)
    ax.grid(which='major', color='#666666', linestyle='-', linewidth=0.4)
    ax.grid(which='minor', color='#999999', linestyle=':', linewidth=0.3)
    ax.minorticks_on()


def main_grid():
    """Create a single figure with a 5x7 grid of subplots.
    Rows = nmu values (5), Columns = sparsity values (7).
    """
    sparsity_l = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
    nmu_l = [10, 100, 1000, 10000, 100000]

    n_rows = len(nmu_l)       # 5
    n_cols = len(sparsity_l)  # 7

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * 4, 5 * 3.5),
                             sharex=True, sharey=True)

    # Preload dense baseline once (doesn't depend on sparsity/nmu)
    dense_template = build_path_template("dense")
    dense_mean, dense_std = load_and_aggregate(dense_template)

    # Set pruning ratio (held constant at 0.3 for all runs)
    set_param("pruning_ratio", 0.3)

    for row, nmu_val in enumerate(nmu_l):
        set_param("nmu", nmu_val)
        for col, sparsity_val in enumerate(sparsity_l):
            set_param("sparsity", sparsity_val)

            ax = axes[row, col]
            plot_subplot(ax, dense_mean, dense_std)

            # Column headers (top row only)
            if row == 0:
                ax.set_title(f"Sparsity: {sparsity_val}", fontsize=10, fontweight='bold')

            # Row labels (leftmost column only)
            if col == 0:
                ax.set_ylabel(f"NMU: {nmu_val}\nTest Accuracy (%)", fontsize=9)

            # X-axis label (bottom row only)
            if row == n_rows - 1:
                ax.set_xlabel("Steps", fontsize=9)

    # Single shared legend from the first subplot that has data
    handles, leg_labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, leg_labels, loc='upper center', ncol=len(leg_labels),
                   fontsize=10, frameon=True, bbox_to_anchor=(0.5, 1.0))

    fig.suptitle(f"Pruning Ratio: {DEF_PRUNING_RATIO}  |  Rows: NMU  |  Columns: Sparsity",
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig("plotter/plots/full_grid_pr0.3.svg", bbox_inches='tight')
    plt.savefig("plotter/plots/full_grid_pr0.3.png", bbox_inches='tight', dpi=150)
    plt.show()


# --- Old single-plot main() (commented out as fallback) ---
# def main(zoom=False):
#     global X_LIM, Y_LIM
#     if zoom:
#         X_LIM = (700, 1000)
#         Y_LIM = (0.7, 0.9)
#         zoom_text = "_Zoomed"
#     else:
#         X_LIM = (0, 1000)
#         Y_LIM = (0, 0.9)
#         zoom_text = ""
#
#     plt.figure(figsize=(20, 8))
#
#     dense_template = build_path_template("dense")
#     dense_mean, dense_std = load_and_aggregate(dense_template)
#     if dense_mean is not None:
#         x = np.arange(len(dense_mean))
#         plt.plot(x, dense_mean, label="Dense", color='black', linestyle='-', linewidth=0.4, alpha=0.7, zorder=10)
#         plt.fill_between(x, dense_mean - dense_std, dense_mean + dense_std, color='black', alpha=0.2, zorder=0)
#
#     sparsifier_names = ["rigl", "set", "gmp", "static"]
#     for i, name in enumerate(sparsifier_names):
#         template = build_path_template(name)
#         mean, std = load_and_aggregate(template)
#         if mean is not None:
#             x = np.arange(len(mean))
#             plt.plot(x, mean, label=labels[i], color=colors[i], linestyle='-', linewidth=0.3, alpha=0.7, zorder=8)
#             plt.fill_between(x, mean - std, mean + std, color=colors[i], alpha=0.2, zorder=2)
#
#     plt.legend()
#     plt.grid(True)
#     plt.minorticks_on()
#     plt.xlim(X_LIM)
#     plt.ylim(Y_LIM)
#     plt.grid(which='major', color='#666666', linestyle='-', linewidth=0.8)
#     plt.grid(which='minor', color='#999999', linestyle=':', linewidth=0.5)
#     plt.xlabel("Steps")
#     plt.ylabel("Test Accuracy (%)")
#     plt.title(f"Sparsity: {DEF_SPARSITY}  Pruning Ratio: {DEF_PRUNING_RATIO}  NMU: {NMU}")
#     plt.savefig(f"plotter/plots/plot_s{DEF_SPARSITY}_pr{DEF_PRUNING_RATIO}_nmu{NMU}{zoom_text}.svg")
#     plt.show()


if __name__ == "__main__":
    main_grid()

