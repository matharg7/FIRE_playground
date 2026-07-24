
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

DEF_SPARSITY, DEF_PRUNING_RATIO, DEF_DT, DEF_GDT = 0, 0, 0, 0
X_LIM = (0, 1000)
Y_LIM = (0, 0.9)

# Clean dictionary-based nmu -> (dt, gdt) mapping
NMU_TO_DT = {
    10000: (1, 1),
    1000:  (8, 6),
    100:   (86, 64),
    10:    (864, 648),
    1:     (8640, 6480),
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
    global DEF_SPARSITY, DEF_PRUNING_RATIO, DEF_DT, DEF_GDT
    if param == "sparsity":
        DEF_SPARSITY = value
    elif param == "pruning_ratio":
        DEF_PRUNING_RATIO = value
    elif param == "nmu":
        if value not in NMU_TO_DT:
            print(f"Warning: nmu={value} not in NMU_TO_DT mapping, defaulting to nmu=1")
            value = 1
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


def main(zoom=False):
    global X_LIM, Y_LIM
    if zoom:
        X_LIM = (700, 1000)
        Y_LIM = (0.7, 0.9)
        zoom_text = "_Zoomed"
    else:
        X_LIM = (0, 1000)
        Y_LIM = (0, 0.9)
        zoom_text = ""

    plt.figure(figsize=(20, 8))

    # --- Dense baseline (mean ± std) ---
    dense_template = build_path_template("dense")
    dense_mean, dense_std = load_and_aggregate(dense_template)
    if dense_mean is not None:
        x = np.arange(len(dense_mean))
        plt.plot(x, dense_mean, label="Dense", color='black', linestyle='-', linewidth=2.5, alpha=0.7, zorder=10)
        plt.fill_between(x, dense_mean - dense_std, dense_mean + dense_std, color='black', alpha=0.2, zorder=9)

    # --- Sparsifiers (mean ± std) ---
    sparsifier_names = ["rigl", "set", "gmp", "static"]
    for i, name in enumerate(sparsifier_names):
        template = build_path_template(name)
        mean, std = load_and_aggregate(template)
        if mean is not None:
            x = np.arange(len(mean))
            plt.plot(x, mean, label=labels[i], color=colors[i], linestyle='-', linewidth=1, alpha=0.7, zorder=1)
            plt.fill_between(x, mean - std, mean + std, color=colors[i], alpha=0.2, zorder=0)

    plt.legend()
    plt.grid(True)
    plt.minorticks_on()
    plt.xlim(X_LIM)
    plt.ylim(Y_LIM)

    plt.grid(which='major', color='#666666', linestyle='-', linewidth=0.8)
    plt.grid(which='minor', color='#999999', linestyle=':', linewidth=0.5)
    plt.xlabel("Steps")
    plt.ylabel("Test Accuracy (%)")

    plt.title(
        f"Sparsity: {DEF_SPARSITY}  Pruning Ratio: {DEF_PRUNING_RATIO}  Delta t: {DEF_DT}"
    )
    plt.savefig(
        f"plot_s{DEF_SPARSITY}_pr{DEF_PRUNING_RATIO}_dt{DEF_DT}{zoom_text}.png"
    )

    plt.show()


if __name__ == "__main__":
    set_param("sparsity", 0.7)
    set_param("pruning_ratio", 0.9)
    set_param("nmu", 10000)

    main()
    main(True)

    set_params(0.7, 0.8, 1000)
    main()
    main(True)

# RESNET18_CIFAR10_gmp_accel_sparsity_0.0_final_sparsity_0.99_delta_t_6480_sparsifier_gmp_seed0.csv
# RESNET18_CIFAR10_gmp_accel_sparsity_0.0_final_sparsity_0.7_delta_t_8_sparsifier_gmp_seed0.csv