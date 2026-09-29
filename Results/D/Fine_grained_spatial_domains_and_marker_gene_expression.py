"""Render the fine-grained spatial-domain and marker-expression panel.

The three CSV files shipped beside this script contain the spatial metadata,
marker statistics and enrichment mapping. The only dataset input is the E18
RNA ``.h5ad`` file, which should be downloaded with the public data package.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_SPATIAL = SCRIPT_DIR / "source_spatial_four_panel.csv"
DEFAULT_CANDIDATES = SCRIPT_DIR / "candidates_rawP0.05_LFC0.25.csv"
DEFAULT_TERMS = SCRIPT_DIR / "metascape_significant_terms_q0.05.csv"
DEFAULT_RNA = REPO_ROOT / "data" / "Mouse_Brain_E18_MISAR" / "adata_RNA.h5ad"
DEFAULT_OUTPUT = SCRIPT_DIR / "Fine_grained_spatial_domains_and_marker_gene_expression.png"

D01_COLOR = "#C08B36"
D06_COLOR = "#397C75"
BACKGROUND = "#E5E7E4"
INK = "#252827"

GENE_SPECS = [
    {"gene": "Htr1f", "domain": "D01", "term_id": "GO:0007268"},
    {"gene": "Wnt7b", "domain": "D01", "term_id": "GO:0050808"},
    {"gene": "Prickle1", "domain": "D01", "term_id": "GO:0034330"},
    {"gene": "Sema7a", "domain": "D06", "term_id": "GO:0031346"},
    {"gene": "Itpr1", "domain": "D06", "term_id": "mmu04540"},
    {"gene": "Cnr1", "domain": "D06", "term_id": "GO:0010975"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spatial", type=Path, default=DEFAULT_SPATIAL)
    parser.add_argument("--rna", type=Path, default=DEFAULT_RNA)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--terms", type=Path, default=DEFAULT_TERMS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def collapse_duplicate_features(
    matrix: sparse.spmatrix, names: pd.Index
) -> tuple[sparse.csr_matrix, pd.Index]:
    names = pd.Index(names.astype(str))
    codes, unique = pd.factorize(names, sort=False)
    matrix = matrix.tocsr().astype(np.float32)
    if len(unique) == len(names):
        return matrix, pd.Index(unique)
    mapper = sparse.csr_matrix(
        (np.ones(len(codes), dtype=np.float32),
         (np.arange(len(codes)), codes)),
        shape=(len(codes), len(unique)),
    )
    return (matrix @ mapper).tocsr(), pd.Index(unique)


def validate_sources(
    metadata: pd.DataFrame, candidates: pd.DataFrame, terms: pd.DataFrame
) -> None:
    required_metadata = {
        "barcode", "x", "y", "haf_domain", "case_domain", "reference_domain"
    }
    missing = required_metadata - set(metadata.columns)
    if missing:
        raise ValueError(f"Spatial metadata is missing columns: {sorted(missing)}")
    for spec in GENE_SPECS:
        marker = candidates.loc[candidates["gene"].eq(spec["gene"])]
        if len(marker) != 1 or marker.iloc[0]["higher_in"] != spec["domain"]:
            raise ValueError(f"Invalid sensitivity-retained marker: {spec['gene']}")
        term = terms.loc[
            terms["GeneList"].eq(spec["domain"]) & terms["GO"].eq(spec["term_id"])
        ]
        if len(term) != 1:
            raise ValueError(f"Missing significant term {spec['term_id']} for {spec['domain']}")
        if spec["gene"] not in str(term.iloc[0]["Hits"]).split("|"):
            raise ValueError(f"{spec['gene']} is not mapped to term {spec['term_id']}")


def load_expression(
    rna_path: Path, metadata: pd.DataFrame, selected_genes: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = sc.read_h5ad(rna_path)
    raw_barcodes = raw.obs_names.astype(str).to_numpy()
    metadata_barcodes = metadata["barcode"].astype(str).to_numpy()
    if not np.array_equal(raw_barcodes, metadata_barcodes):
        raise ValueError("RNA and spatial metadata spot orders differ")

    counts, genes = collapse_duplicate_features(raw.X, raw.var_names)
    gene_to_index = {gene: index for index, gene in enumerate(genes)}
    missing = [gene for gene in selected_genes if gene not in gene_to_index]
    if missing:
        raise ValueError("Selected genes are absent from RNA data: " + ", ".join(missing))
    library_sizes = np.asarray(counts.sum(axis=1)).ravel().astype(float)
    if np.any(library_sizes <= 0):
        raise ValueError("RNA matrix contains an empty spot")

    selected_counts = counts[:, [gene_to_index[gene] for gene in selected_genes]]
    normalized = selected_counts.multiply((10_000.0 / library_sizes)[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)
    values = normalized.toarray().astype(float)

    wide = metadata[["barcode", "x", "y", "haf_domain", "case_domain"]].copy()
    for index, gene in enumerate(selected_genes):
        wide[gene] = values[:, index]
    long = wide.melt(
        id_vars=["barcode", "x", "y", "haf_domain", "case_domain"],
        value_vars=selected_genes,
        var_name="gene",
        value_name="log1p_counts_per_10000",
    )
    return wide, long


def style_spatial_axis(axis: plt.Axes, metadata: pd.DataFrame) -> None:
    x_range = float(metadata["x"].max() - metadata["x"].min())
    y_range = float(metadata["y"].max() - metadata["y"].min())
    axis.set_xlim(metadata["x"].min() - 0.018 * x_range,
                  metadata["x"].max() + 0.018 * x_range)
    axis.set_ylim(metadata["y"].min() - 0.018 * y_range,
                  metadata["y"].max() + 0.018 * y_range)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def expression_colormap(domain: str, name: str) -> LinearSegmentedColormap:
    colors = {
        "D01": ["#F2F1EC", "#E6DABF", "#D2B06B", "#B27A2E", "#744A16"],
        "D06": ["#F1F2F0", "#D8E4E0", "#9FC5BD", "#4F9188", "#1F5F5A"],
    }
    return LinearSegmentedColormap.from_list(name, colors[domain], N=256)


def spot_size(figure: plt.Figure, axis: plt.Axes) -> float:
    origin = np.array([axis.get_xlim()[0], axis.get_ylim()[0]])
    projected = axis.transData.transform(
        np.vstack([origin, origin + [1, 0], origin + [0, 1]])
    )
    pitch = min(
        np.linalg.norm(projected[1] - projected[0]),
        np.linalg.norm(projected[2] - projected[0]),
    )
    return float((0.95 * pitch * 72 / figure.dpi) ** 2)


def draw_domain_panel(
    axis: plt.Axes, metadata: pd.DataFrame, domain: str, color: str, size: float
) -> None:
    axis.scatter(metadata["x"], metadata["y"], s=size, color=BACKGROUND,
                 linewidths=0, rasterized=True)
    mask = metadata["case_domain"].eq(domain)
    axis.scatter(metadata.loc[mask, "x"], metadata.loc[mask, "y"], s=size,
                 color=color, linewidths=0, rasterized=True)
    style_spatial_axis(axis, metadata)


def draw_expression_panel(
    axis: plt.Axes, metadata: pd.DataFrame, values: np.ndarray,
    cmap: mpl.colors.Colormap, norm: Normalize, size: float
) -> None:
    order = np.argsort(values, kind="stable")
    axis.scatter(metadata["x"].to_numpy()[order], metadata["y"].to_numpy()[order],
                 c=values[order], s=size, cmap=cmap, norm=norm,
                 linewidths=0, rasterized=True)
    style_spatial_axis(axis, metadata)


def main() -> None:
    args = parse_args()
    paths = {
        "spatial metadata": args.spatial,
        "RNA dataset": args.rna,
        "candidate table": args.candidates,
        "enrichment table": args.terms,
    }
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing D-panel input(s):\n" + "\n".join(missing))

    metadata = pd.read_csv(args.spatial.resolve())
    candidates = pd.read_csv(args.candidates.resolve())
    terms = pd.read_csv(args.terms.resolve())
    validate_sources(metadata, candidates, terms)

    selected_genes = [spec["gene"] for spec in GENE_SPECS]
    wide, long = load_expression(args.rna.resolve(), metadata, selected_genes)
    dpallm_mask = metadata["reference_domain"].eq("DPallm").to_numpy()
    metadata = metadata.loc[dpallm_mask].copy()
    values = wide.loc[dpallm_mask].copy()
    long = long.loc[long["barcode"].isin(metadata["barcode"])]

    norm = Normalize(
        vmin=0.0,
        vmax=float(np.ceil(np.quantile(values[selected_genes].to_numpy(dtype=float), 0.99) * 10) / 10),
        clip=True,
    )
    cmaps = {
        domain: expression_colormap(domain, f"results_d_{domain}")
        for domain in ("D01", "D06")
    }

    figure = plt.figure(figsize=(7.2, 3.15), dpi=180)
    lefts = [0.022, 0.268, 0.514, 0.760]
    width = 0.175
    height = 0.345
    bottoms = {"D01": 0.555, "D06": 0.095}
    rows = [("D01", D01_COLOR), ("D06", D06_COLOR)]

    for domain, domain_color in rows:
        axes = []
        for left in lefts:
            axis = figure.add_axes([left, bottoms[domain], width, height])
            style_spatial_axis(axis, metadata)
            axes.append(axis)
        figure.canvas.draw()
        size = spot_size(figure, axes[0])
        draw_domain_panel(axes[0], metadata, domain, domain_color, size)
        genes = [spec["gene"] for spec in GENE_SPECS if spec["domain"] == domain]
        for axis, gene in zip(axes[1:], genes):
            draw_expression_panel(axis, metadata, values[gene].to_numpy(),
                                  cmaps[domain], norm, size)

        top = bottoms[domain] + height
        titles = [f"{domain} spatial domain", *genes]
        for axis, title in zip(axes, titles):
            figure.text(axis.get_position().x0 + axis.get_position().width / 2,
                        top + 0.018, title, ha="center", va="bottom",
                        fontsize=10.0 if title.startswith("D") else 9.0,
                        fontweight="semibold", color=INK)

        colorbar_axis = figure.add_axes(
            [lefts[-1] + width + 0.012, bottoms[domain] + 0.015,
             0.012, height - 0.03]
        )
        colorbar = mpl.colorbar.ColorbarBase(colorbar_axis, cmap=cmaps[domain], norm=norm)
        colorbar.set_ticks([0.0, norm.vmax / 2.0, norm.vmax])
        colorbar.set_ticklabels(["0", f"{norm.vmax / 2:.1f}", f"{norm.vmax:.1f}"])
        colorbar.ax.tick_params(labelsize=7.0, length=2.0, pad=2.0)
        colorbar.outline.set_linewidth(0.8)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
