"""Paper-ready Plotly plots for the exported continual-learning runs."""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_ROOT = os.path.join(BASE_DIR, "csv_export")
PLOT_DIR = os.path.join(BASE_DIR, "plotter", "plots", "plotly")
os.makedirs(PLOT_DIR, exist_ok=True)

SEEDS = [0, 5, 8]
DISPLAY_SMOOTHING = 1
SD_BAND_ALPHA = 0.24
SD_EDGE_ALPHA = 0.65
X_LIM = (0, 1000)
Y_LIM = (0.1, 0.35)

COLORS = {
    "dense": "#222222",
    "rigl": "#D55E00",
    "set": "#0072B2",
    "gmp": "#009E73",
    "static": "#CC79A7",
}
LABELS = {
    "dense": "Dense",
    "rigl": "RigL",
    "set": "SET",
    "gmp": "GMP",
    "static": "Static",
}
LINE_WIDTHS = {"dense": 1.4, "rigl": 0.8, "set": 0.8,
               "gmp": 0.8, "static": 0.8}


def _dense_path(task, seed):
    return os.path.join(CSV_ROOT, task, "dense", f"dense_seed{seed}.csv")


def _static_path(task, seed, sparsity):
    return os.path.join(CSV_ROOT, task, "static",
                        f"static_seed{seed}_sp{sparsity}.csv")


def _gmp_path(task, seed, sparsity, nmu):
    return os.path.join(CSV_ROOT, task, "gmp",
                        f"gmp_seed{seed}_sp{sparsity}_nmu{nmu}.csv")


def _dst_path(task, method, seed, sparsity, pruning_ratio, nmu):
    return os.path.join(CSV_ROOT, task, method,
                        f"{method}_seed{seed}_sp{sparsity}"
                        f"_pr{pruning_ratio}_nmu{nmu}.csv")


def load_and_aggregate(path_fn):
    """Return mean and standard deviation across available seed runs."""
    series = []
    for seed in SEEDS:
        filepath = path_fn(seed)
        if os.path.isfile(filepath):
            frame = pd.read_csv(filepath)
            if "accuracy" not in frame:
                raise ValueError(f"Missing 'accuracy' column in {filepath}")
            series.append(frame["accuracy"].to_numpy())
    if not series:
        return None, None
    length = min(len(values) for values in series)
    stacked = np.stack([values[:length] for values in series])
    return stacked.mean(axis=0), stacked.std(axis=0)


def _smooth(values):
    if DISPLAY_SMOOTHING <= 1:
        return np.asarray(values)
    return (pd.Series(values)
            .rolling(DISPLAY_SMOOTHING, center=True, min_periods=1)
            .mean().to_numpy())


def _rgba(hex_color, alpha):
    rgb = tuple(int(hex_color[index:index + 2], 16) for index in (1, 3, 5))
    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})"


def _add_series(fig, mean, std, method, row, col, showlegend):
    x = np.arange(len(mean))
    mean, std = _smooth(mean), _smooth(std)
    label, color = LABELS[method], COLORS[method]
    group = method
    fig.add_trace(go.Scatter(
        x=x, y=mean - std, mode="lines",
        line=dict(color=_rgba(color, SD_EDGE_ALPHA), width=0.7,
                  dash="dot"),
        hoverinfo="skip", showlegend=False, legendgroup=group,
    ), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=x, y=mean + std, mode="lines",
        line=dict(color=_rgba(color, SD_EDGE_ALPHA), width=0.7,
                  dash="dot"),
        fill="tonexty", fillcolor=_rgba(color, SD_BAND_ALPHA),
        hoverinfo="skip",
        showlegend=False, legendgroup=group,
    ), row=row, col=col)
    fig.add_trace(go.Scatter(
        x=x, y=mean, mode="lines", name=label,
        line=dict(color=color, width=LINE_WIDTHS[method]),
        showlegend=showlegend, legendgroup=group,
        customdata=np.column_stack((std, mean - std, mean + std)),
        hovertemplate=(f"{label}: %{{y:.3f}}"
                       "<br>SD: %{customdata[0]:.3f}"
                       "<br>range: %{customdata[1]:.3f} to "
                       "%{customdata[2]:.3f}<extra></extra>"),
    ), row=row, col=col)


def _add_subplot_data(fig, task, sparsity, nmu, pruning_ratio,
                      dense_mean, dense_std, row, col, showlegend):
    series = [
        ("dense", dense_mean, dense_std),
        ("static", *load_and_aggregate(
            lambda seed: _static_path(task, seed, sparsity))),
        ("gmp", *load_and_aggregate(
            lambda seed: _gmp_path(task, seed, sparsity, nmu))),
    ]
    for method in ("rigl", "set"):
        series.append((method, *load_and_aggregate(
            lambda seed, method=method: _dst_path(
                task, method, seed, sparsity, pruning_ratio, nmu))))
    for method, mean, std in series:
        if mean is not None:
            _add_series(fig, mean, std, method, row, col, showlegend)


def _style(fig, title):
    fig.update_layout(
        template="simple_white",
        title=dict(text=title, x=0.5, xanchor="center",
                   font=dict(size=17, color="#222222")),
        font=dict(family="Arial, sans-serif", size=11, color="#222222"),
        paper_bgcolor="white", plot_bgcolor="white", hovermode="x unified",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.04,
                yanchor="bottom", bgcolor="rgba(255,255,255,0)",
                title=dict(text="Mean +/- 1 SD")),
        margin=dict(l=75, r=25, t=100, b=65),
    )
    fig.update_xaxes(range=list(X_LIM), showline=True, linewidth=1,
                     linecolor="#444444", mirror=True, showgrid=True,
                     gridcolor="#E5E7EB", zeroline=False, ticks="outside")
    fig.update_yaxes(range=list(Y_LIM), showline=True, linewidth=1,
                     linecolor="#444444", mirror=True, showgrid=True,
                     gridcolor="#E5E7EB", zeroline=False, ticks="outside")


def _write_outputs(fig, output_stem):
    fig.write_html(f"{output_stem}.html", include_plotlyjs="cdn")
    try:
        fig.write_image(f"{output_stem}.svg", width=2200, height=1400, scale=2)
        fig.write_image(f"{output_stem}.pdf", width=2200, height=1400, scale=2)
    except (ImportError, RuntimeError, ValueError) as error:
        print("SVG/PDF export skipped: install plotly and kaleido with "
              "'pip install -r plotter/requirements-plotly.txt'.")
        print(f"Kaleido error: {error}")


def main_grid(task="CIFAR100", pruning_ratio=0.3,
              sparsity_list=None, nmu_list=None):
    """Create an NMU-by-sparsity grid with shared axes."""
    sparsity_list = sparsity_list or [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
    nmu_list = nmu_list or [10, 100, 1000, 10000, 100000]
    rows, cols = len(nmu_list), len(sparsity_list)
    fig = make_subplots(
        rows=rows, cols=cols, shared_xaxes=True, shared_yaxes=True,
        subplot_titles=[f"Sparsity = {value:g}" for value in sparsity_list],
        horizontal_spacing=0.025, vertical_spacing=0.045,
    )
    dense_mean, dense_std = load_and_aggregate(lambda seed: _dense_path(task, seed))
    for row, nmu in enumerate(nmu_list, start=1):
        for col, sparsity in enumerate(sparsity_list, start=1):
            _add_subplot_data(fig, task, sparsity, nmu, pruning_ratio,
                              dense_mean, dense_std, row, col,
                              row == 1 and col == 1)
            if col == 1:
                fig.update_yaxes(title_text=f"NMU {nmu:g}", row=row, col=col)
    fig.update_xaxes(title_text="Training steps", row=rows, col=1)
    fig.update_yaxes(title_text="Test accuracy", row=1, col=1)
    _style(fig, f"{task} | pruning ratio {pruning_ratio:g}")
    _write_outputs(fig, os.path.join(PLOT_DIR, f"{task}_grid_pr{pruning_ratio}"))
    return fig


def main(task="CIFAR10", sparsity=0.9, pruning_ratio=0.3, nmu=10000,
         zoom=False):
    """Create one paper-ready panel and return its Plotly figure."""
    fig = make_subplots(rows=1, cols=1)
    dense_mean, dense_std = load_and_aggregate(lambda seed: _dense_path(task, seed))
    _add_subplot_data(fig, task, sparsity, nmu, pruning_ratio,
                      dense_mean, dense_std, 1, 1, True)
    fig.update_xaxes(title_text="Training steps")
    fig.update_yaxes(title_text="Test accuracy")
    _style(fig, f"{task} | sparsity {sparsity:g} | pruning ratio {pruning_ratio:g} | NMU {nmu:g}")
    if zoom:
        fig.update_xaxes(range=[700, 1000])
        fig.update_yaxes(range=[0.7, 0.9])
    output_stem = os.path.join(
        PLOT_DIR, f"{task}_s{sparsity}_pr{pruning_ratio}_nmu{nmu}")
    _write_outputs(fig, output_stem)
    return fig


if __name__ == "__main__":
    for pruning_ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
        main_grid(task="CIFAR100", pruning_ratio=pruning_ratio)
