import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch

from .datasets import DatasetSpec
from .evaluation import clustering, compute_external_metrics
from .preprocess import (
    clr_normalize_each_cell,
    construct_neighbor_graph,
    fix_seed,
    lsi,
    pca,
)
from .trainer import HiFuseTrainer
from .visualization import save_comparison


BARCODE_COLUMN = "Barcode"
LABEL_COLUMN = "manual-anno"
EMBEDDING_DIMENSION = 64


def _clean_labels(values):
    values = pd.Series(values, dtype="string").str.strip()
    return values.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})


def _read_obs_names(path):
    adata = ad.read_h5ad(path, backed="r")
    try:
        return pd.Index(adata.obs_names.astype(str))
    finally:
        adata.file.close()


def _read_obs(path, columns):
    adata = ad.read_h5ad(path, backed="r")
    try:
        frame = adata.obs.loc[:, list(columns)].copy()
        frame.index = frame.index.astype(str)
        return frame
    finally:
        adata.file.close()


def _read_text_labels(path):
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    return _clean_labels(lines)


def _align_numeric_barcodes(obs_names, labels):
    if len(obs_names) != len(labels):
        raise ValueError(
            f"Label length mismatch: expected {len(obs_names)}, found {len(labels)}"
        )
    barcode_frame = pd.DataFrame({BARCODE_COLUMN: obs_names.astype(str)})
    barcode_frame["numeric"] = pd.to_numeric(barcode_frame[BARCODE_COLUMN], errors="raise")
    barcode_frame["original_position"] = range(len(barcode_frame))
    barcode_frame = barcode_frame.sort_values(
        ["numeric", BARCODE_COLUMN],
        kind="mergesort",
    ).reset_index(drop=True)
    aligned = pd.Series(pd.NA, index=range(len(barcode_frame)), dtype="string")
    aligned.iloc[barcode_frame["original_position"].to_numpy()] = labels.to_numpy()
    return pd.DataFrame(
        {
            BARCODE_COLUMN: obs_names.astype(str),
            LABEL_COLUMN: aligned,
        }
    )


def _align_partial_reference(rna_path, obs_names, labels, reference_column):
    adata = ad.read_h5ad(rna_path, backed="r")
    try:
        if reference_column not in adata.obs:
            raise ValueError(f"{reference_column} was not found in {rna_path}")
        reference = _clean_labels(
            adata.obs[reference_column].astype(str).str.replace("C", "", regex=False)
        ).reset_index(drop=True)
    finally:
        adata.file.close()

    aligned = pd.Series(pd.NA, index=range(len(reference)), dtype="string")
    label_position = 0
    for reference_position in range(len(reference)):
        if label_position >= len(labels):
            break
        if reference.iloc[reference_position] == labels.iloc[label_position]:
            aligned.iloc[reference_position] = labels.iloc[label_position]
            label_position += 1
            continue
        next_reference_matches = (
            reference_position + 1 < len(reference)
            and reference.iloc[reference_position + 1] == labels.iloc[label_position]
        )
        if next_reference_matches:
            continue
        raise ValueError(
            "Unable to align partial labels at reference position "
            f"{reference_position} and label position {label_position}"
        )
    if label_position != len(labels):
        raise ValueError(f"Only aligned {label_position} of {len(labels)} labels")
    return pd.DataFrame(
        {
            BARCODE_COLUMN: obs_names.astype(str),
            LABEL_COLUMN: aligned,
        }
    )


def prepare_annotation(dataset_dir, spec, obs_names, annotation_dir):
    label_spec = spec.labels
    label_path = dataset_dir / label_spec.filename
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing annotation source: {label_path}")

    if label_spec.kind == "csv":
        source = pd.read_csv(label_path)
        annotation = pd.DataFrame(
            {
                BARCODE_COLUMN: source[label_spec.barcode_column].astype(str),
                LABEL_COLUMN: _clean_labels(source[label_spec.label_column]),
            }
        )
        note = "annotation csv"
    elif label_spec.kind == "h5ad_obs":
        obs = _read_obs(label_path, label_spec.obs_columns)
        selected_column = None
        selected_labels = None
        for column in label_spec.obs_columns:
            candidate = _clean_labels(obs[column])
            if candidate.notna().any():
                selected_column = column
                selected_labels = candidate
                break
        if selected_labels is None:
            raise ValueError(f"No usable annotation column found in {label_path}")
        annotation = pd.DataFrame(
            {
                BARCODE_COLUMN: obs.index.astype(str),
                LABEL_COLUMN: selected_labels,
            }
        )
        note = f"h5ad obs column {selected_column}"
    elif label_spec.kind == "text":
        labels = _read_text_labels(label_path)
        if label_spec.numeric_barcode_order:
            annotation = _align_numeric_barcodes(obs_names, labels)
            note = "text labels aligned by numeric barcode order"
        elif label_spec.partial_reference_column:
            annotation = _align_partial_reference(
                dataset_dir / spec.rna_filename,
                obs_names,
                labels,
                label_spec.partial_reference_column,
            )
            note = (
                f"text labels aligned to {labels.notna().sum()} of {len(obs_names)} spots; "
                f"unlabeled spots={annotation[LABEL_COLUMN].isna().sum()}"
            )
        else:
            if len(labels) != len(obs_names):
                raise ValueError(
                    f"Label length mismatch for {label_path}: "
                    f"expected {len(obs_names)}, found {len(labels)}"
                )
            annotation = pd.DataFrame(
                {
                    BARCODE_COLUMN: obs_names.astype(str),
                    LABEL_COLUMN: labels,
                }
            )
            note = "text labels aligned by spot order"
    else:
        raise ValueError(f"Unsupported annotation kind: {label_spec.kind}")

    annotation[BARCODE_COLUMN] = annotation[BARCODE_COLUMN].astype(str).str.strip()
    annotation[LABEL_COLUMN] = _clean_labels(annotation[LABEL_COLUMN])
    annotation_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = annotation_dir / f"{spec.name}.csv"
    annotation.to_csv(annotation_path, index=False)
    cluster_count = int(annotation[LABEL_COLUMN].dropna().nunique())
    return annotation, annotation_path, cluster_count, note


def _prepare_rna(adata):
    adata = adata.copy()
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat_v3")
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    components = min(100, adata.n_obs - 1, adata.n_vars - 1)
    adata.obsm["feat"] = pca(adata, n_comps=components)
    return adata


def _prepare_secondary(adata, modality):
    adata = adata.copy()
    if modality == "adt":
        clr_normalize_each_cell(adata)
        adata.obsm["feat"] = (
            adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
        )
        return adata
    if modality == "atac":
        if "X_lsi" not in adata.obsm:
            components = min(100, adata.n_obs - 1, adata.n_vars - 1)
            lsi(adata, n_components=components)
        adata.obsm["feat"] = np.asarray(adata.obsm["X_lsi"])
        return adata
    raise ValueError(f"Unsupported secondary modality: {modality}")


def _resolve_device(device_name):
    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA is unavailable; using CPU instead of {device_name}", flush=True)
        return torch.device("cpu")
    return torch.device(device_name)


def run_dataset(
    spec: DatasetSpec,
    data_root,
    output_root,
    device_name="cuda:0",
    seed=2022,
    make_plots=True,
):
    data_root = Path(data_root)
    output_root = Path(output_root)
    dataset_dir = data_root / spec.name
    output_dir = output_root / spec.name
    output_dir.mkdir(parents=True, exist_ok=True)

    rna_path = dataset_dir / spec.rna_filename
    secondary_path = dataset_dir / spec.secondary_filename
    for path in (rna_path, secondary_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing dataset file: {path}")

    obs_names = _read_obs_names(rna_path)
    annotation, annotation_path, cluster_count, annotation_note = prepare_annotation(
        dataset_dir,
        spec,
        obs_names,
        output_root / "prepared_annotations",
    )

    fix_seed(seed)
    print("  reading and preprocessing modalities", flush=True)
    adata_rna = _prepare_rna(sc.read_h5ad(rna_path))
    adata_secondary = _prepare_secondary(
        sc.read_h5ad(secondary_path),
        spec.secondary_type,
    )
    graph_data = construct_neighbor_graph(
        adata_rna,
        adata_secondary,
        datatype=spec.platform,
        n_neighbors=spec.spatial_neighbors,
        feature_k=spec.feature_neighbors,
    )
    device = _resolve_device(device_name)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    trainer = HiFuseTrainer(
        data=graph_data,
        datatype=spec.platform,
        device=device,
        random_seed=seed,
        epochs=spec.epochs,
        dim_output=EMBEDDING_DIMENSION,
        pretrain_ratio=spec.pretrain_ratio,
    )
    print(f"  training for {spec.epochs} epochs on {device}", flush=True)
    result = trainer.train()

    print(f"  clustering {cluster_count} spatial domains with mclust", flush=True)
    adata_rna.obsm["HiFuse"] = result["HiFuse"]
    clustering(
        adata_rna,
        n_clusters=cluster_count,
        key="HiFuse",
        add_key="HiFuse_cluster",
        method="mclust",
        use_pca=True,
        n_comps=min(spec.cluster_pca_dims, result["HiFuse"].shape[1]),
        mclust_model=spec.mclust_model,
    )

    adata_rna.obs[BARCODE_COLUMN] = adata_rna.obs_names.astype(str)
    predictions = adata_rna.obs[[BARCODE_COLUMN, "HiFuse_cluster"]].copy()
    predictions = predictions.merge(annotation, on=BARCODE_COLUMN, how="left")
    valid = predictions[LABEL_COLUMN].notna()
    metrics = compute_external_metrics(
        predictions.loc[valid, LABEL_COLUMN],
        predictions.loc[valid, "HiFuse_cluster"].astype(str),
    )
    np.save(output_dir / "embedding.npy", result["HiFuse"])
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    if make_plots:
        labels_by_barcode = predictions.set_index(BARCODE_COLUMN)[LABEL_COLUMN]
        adata_rna.obs["manual_anno"] = labels_by_barcode.loc[
            adata_rna.obs[BARCODE_COLUMN]
        ].to_numpy()
        save_comparison(adata_rna, output_dir / "comparison_panel.png")

    return {
        "dataset": spec.name,
        "status": "ok",
        "seed": int(seed),
        "epochs": int(spec.epochs),
        "n_clusters": int(cluster_count),
        "annotation": str(annotation_path),
        "annotation_note": annotation_note,
        "rna": str(rna_path),
        "secondary": str(secondary_path),
        "secondary_type": spec.secondary_type,
        "platform": spec.platform,
        "output": str(output_dir),
        "metrics": metrics,
    }
