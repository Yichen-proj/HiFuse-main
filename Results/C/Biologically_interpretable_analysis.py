"""Render the biologically interpretable analysis panel.

The packaged ``spot_attention.csv`` is the only input required for this
plot. The script does not retrain HiFuse or require the private benchmark
intermediate files used to create that table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "spot_attention.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "Biologically_interpretable_analysis.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def load_frame(path: Path) -> tuple[pd.DataFrame, str, float, float, float, float, float, float]:
    if not path.exists():
        raise FileNotFoundError(f"C-panel input is missing: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"C-panel input is empty: {path}")
    required = {
        "spatial_x",
        "spatial_y",
        "graph_rna_spatial",
        "graph_mod2_spatial",
        "modality_rna",
        "path_nonlinear",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"C-panel input is missing columns: {sorted(missing)}")

    mod2_label = str(frame.get("mod2_type", pd.Series(["ATAC"])).iloc[0]).upper()
    graph_min, graph_max = np.nanquantile(
        frame[["graph_rna_spatial", "graph_mod2_spatial"]].to_numpy(dtype=float),
        [0.02, 0.98],
    )
    modality_min, modality_max = np.nanquantile(
        frame["modality_rna"].to_numpy(dtype=float), [0.02, 0.98]
    )
    path_min, path_max = np.nanquantile(
        frame["path_nonlinear"].to_numpy(dtype=float), [0.02, 0.98]
    )
    return (
        frame,
        mod2_label,
        float(graph_min),
        float(graph_max),
        float(modality_min),
        float(modality_max),
        float(path_min),
        float(path_max),
    )


def spatial_panel(
    figure: plt.Figure,
    axis: plt.Axes,
    frame: pd.DataFrame,
    column: str,
    title: str,
    vmin: float,
    vmax: float,
) -> None:
    values = frame[column].to_numpy(dtype=float)
    order = np.argsort(values)
    point_size = float(np.clip(3900.0 / len(frame), 1.2, 4.0))
    image = axis.scatter(
        frame.loc[order, "spatial_x"],
        frame.loc[order, "spatial_y"],
        c=values[order],
        s=point_size,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
        rasterized=True,
    )
    axis.invert_yaxis()
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(title, fontsize=7.4, fontweight="bold", pad=4)
    for spine in axis.spines.values():
        spine.set_visible(False)
    colorbar = figure.colorbar(
        image,
        ax=axis,
        orientation="horizontal",
        fraction=0.055,
        pad=0.025,
        aspect=18,
    )
    colorbar.set_ticks([vmin, vmax])
    colorbar.set_ticklabels([f"{vmin:.2f}", f"{vmax:.2f}"])
    colorbar.ax.tick_params(labelsize=5.8, length=1.5, pad=1)
    colorbar.outline.set_linewidth(0.5)


def main() -> None:
    args = parse_args()
    (
        frame,
        mod2_label,
        graph_min,
        graph_max,
        modality_min,
        modality_max,
        path_min,
        path_max,
    ) = load_frame(args.input.resolve())

    figure = plt.figure(figsize=(7.2, 2.52))
    grid = figure.add_gridspec(
        1, 4, left=0.035, right=0.995, top=0.94, bottom=0.12, wspace=0.30
    )
    axes = [figure.add_subplot(grid[0, index]) for index in range(4)]
    panels = [
        (
            "graph_rna_spatial",
            "Intra-Modality Attention\nRNA: spatial graph",
            graph_min,
            graph_max,
        ),
        (
            "graph_mod2_spatial",
            f"Intra-Modality Attention\n{mod2_label}: spatial graph",
            graph_min,
            graph_max,
        ),
        (
            "modality_rna",
            "Cross-Modality Attention\nRNA weight",
            modality_min,
            modality_max,
        ),
        (
            "path_nonlinear",
            "Global Attention\nNonlinear path",
            path_min,
            path_max,
        ),
    ]
    for axis, (column, title, vmin, vmax) in zip(axes, panels):
        spatial_panel(figure, axis, frame, column, title, vmin, vmax)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
