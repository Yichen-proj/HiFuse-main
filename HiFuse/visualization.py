from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib import colors as mpl_colors
from scipy.optimize import linear_sum_assignment


BASE_COLORS = (
    "#e07a5f", "#83c5be", "#d3edbe", "#597b84",
    "#f4a259", "#52b69a", "#e6db9e", "#a38161",
    "#f9c74f", "#98df8a", "#ffbb78", "#d5a376",
    "#f19c79", "#6fb7ad", "#c7e7b4", "#7d8b8f",
    "#d88a6d", "#69b99f", "#d9ddb0", "#b08b6b",
    "#ebb38a", "#8fcf95", "#efe0a6", "#c6b39b",
)


def _sort_labels(values):
    def key(value):
        text = str(value)
        return (0, int(text)) if text.isdigit() else (1, text)

    return sorted(values, key=key)


def _palette(labels):
    colors = list(BASE_COLORS)
    if len(labels) > len(colors):
        extras = plt.cm.hsv(np.linspace(0, 1, len(labels) - len(colors), endpoint=False))
        colors.extend(extras)
    return {label: colors[index] for index, label in enumerate(labels)}


def _match_cluster_names(predicted, truth):
    predicted = pd.Series(predicted).astype(str)
    truth = pd.Series(truth).astype("string")
    valid = truth.notna()
    contingency = pd.crosstab(predicted.loc[valid], truth.loc[valid].astype(str))
    if contingency.empty:
        return predicted
    row_indices, column_indices = linear_sum_assignment(-contingency.to_numpy())
    mapping = {
        contingency.index[row]: contingency.columns[column]
        for row, column in zip(row_indices, column_indices)
    }
    return predicted.map(lambda value: mapping.get(value, f"Pred_{value}"))


def _scatter(axis, coordinates, labels, color_map, title, x_label, y_label):
    coordinates = np.asarray(coordinates, dtype=np.float32)
    labels = pd.Series(labels).astype(str)
    for label, color in color_map.items():
        mask = labels == label
        if mask.any():
            axis.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                s=50,
                c=[color],
                linewidths=0,
            )
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)


def save_comparison(adata, output_path):
    output_path = Path(output_path)
    working = adata.copy()
    predicted = _match_cluster_names(
        working.obs["HiFuse_cluster"],
        working.obs["manual_anno"],
    )
    truth = working.obs["manual_anno"].astype("string").fillna("Unlabeled").astype(str)
    labels = _sort_labels(set(predicted.unique()) | set(truth.unique()))
    color_map = _palette(labels)

    sc.pp.neighbors(working, use_rep="HiFuse")
    sc.tl.umap(working, random_state=0)
    spatial = np.asarray(working.obsm["spatial"], dtype=np.float32)
    umap = np.asarray(working.obsm["X_umap"], dtype=np.float32)

    np.savez_compressed(
        output_path.with_name("plot_source.npz"),
        spatial=spatial,
        umap=umap,
        predicted_labels=predicted.to_numpy(dtype=str),
        true_labels=truth.to_numpy(dtype=str),
        categories=np.asarray(labels, dtype=str),
        colors=np.asarray([mpl_colors.to_hex(color_map[label]) for label in labels]),
    )

    figure, axes = plt.subplots(2, 2, figsize=(16, 12))
    _scatter(axes[0, 0], spatial, predicted, color_map, "Predicted spatial domains", "x", "y")
    _scatter(axes[0, 1], spatial, truth, color_map, "Ground-truth spatial domains", "x", "y")
    _scatter(axes[1, 0], umap, predicted, color_map, "Predicted domains in UMAP", "UMAP1", "UMAP2")
    _scatter(axes[1, 1], umap, truth, color_map, "Ground truth in UMAP", "UMAP1", "UMAP2")
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            markerfacecolor=color_map[label],
            markersize=6,
            label=label,
        )
        for label in labels
    ]
    figure.legend(
        handles=handles,
        labels=labels,
        title="Aligned labels",
        loc="center left",
        bbox_to_anchor=(0.92, 0.5),
        fontsize=8,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 0.9, 1))
    figure.savefig(output_path, dpi=250)
    plt.close(figure)
