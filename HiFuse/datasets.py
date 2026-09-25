from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple


@dataclass(frozen=True)
class LabelSpec:
    kind: str
    filename: str
    barcode_column: Optional[str] = None
    label_column: Optional[str] = None
    obs_columns: Tuple[str, ...] = ()
    numeric_barcode_order: bool = False
    partial_reference_column: Optional[str] = None


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    rna_filename: str
    secondary_filename: str
    secondary_type: str
    platform: str
    epochs: int
    spatial_neighbors: int
    feature_neighbors: int
    cluster_pca_dims: int
    pretrain_ratio: float
    labels: LabelSpec
    mclust_model: str = "EEE"


DATASETS = (
    DatasetSpec(
        name="HLN",
        rna_filename="adata_RNA.h5ad",
        secondary_filename="adata_ADT.h5ad",
        secondary_type="adt",
        platform="10x",
        epochs=200,
        spatial_neighbors=7,
        feature_neighbors=25,
        cluster_pca_dims=25,
        pretrain_ratio=0.1,
        labels=LabelSpec(kind="text", filename="GT_labels.txt"),
    ),
    DatasetSpec(
        name="Human_Brain_Hippocampal_Spatial-epigenome-transcriptome",
        rna_filename="Human_RNA.h5ad",
        secondary_filename="Human_ATAC_lsi.h5ad",
        secondary_type="atac",
        platform="Spatial-epigenome-transcriptome",
        epochs=1200,
        spatial_neighbors=10,
        feature_neighbors=20,
        cluster_pca_dims=10,
        pretrain_ratio=0.3,
        labels=LabelSpec(
            kind="h5ad_obs",
            filename="Human_RNA.h5ad",
            obs_columns=("final_annot",),
        ),
    ),
    DatasetSpec(
        name="Human_Lymph_Node_S2",
        rna_filename="adata_RNA.h5ad",
        secondary_filename="adata_ADT.h5ad",
        secondary_type="adt",
        platform="10x",
        epochs=400,
        spatial_neighbors=8,
        feature_neighbors=10,
        cluster_pca_dims=30,
        pretrain_ratio=0.2,
        labels=LabelSpec(
            kind="csv",
            filename="annotation_lymph_node_D1.csv",
            barcode_column="Barcode",
            label_column="manual",
        ),
    ),
    DatasetSpec(
        name="Human_Lymph_Node_S3",
        rna_filename="adata_RNA.h5ad",
        secondary_filename="adata_ADT.h5ad",
        secondary_type="adt",
        platform="10x",
        epochs=800,
        spatial_neighbors=5,
        feature_neighbors=10,
        cluster_pca_dims=25,
        pretrain_ratio=0.1,
        labels=LabelSpec(
            kind="h5ad_obs",
            filename="adata_RNA.h5ad",
            obs_columns=("final_annot",),
        ),
        mclust_model="VVV",
    ),
    DatasetSpec(
        name="Human_tonsil_s1_10x",
        rna_filename="adata_rna.h5ad",
        secondary_filename="adata_adt.h5ad",
        secondary_type="adt",
        platform="10x",
        epochs=100,
        spatial_neighbors=3,
        feature_neighbors=30,
        cluster_pca_dims=20,
        pretrain_ratio=0.1,
        labels=LabelSpec(
            kind="h5ad_obs",
            filename="adata_rna.h5ad",
            obs_columns=("final_annot",),
        ),
    ),
    DatasetSpec(
        name="Mouse_Brain",
        rna_filename="adata_RNA.h5ad",
        secondary_filename="adata_peaks_normalized.h5ad",
        secondary_type="atac",
        platform="Spatial-epigenome-transcriptome",
        epochs=200,
        spatial_neighbors=7,
        feature_neighbors=25,
        cluster_pca_dims=15,
        pretrain_ratio=0.0,
        labels=LabelSpec(
            kind="text",
            filename="MB_cluster.txt",
            partial_reference_column="ATAC_clusters",
        ),
    ),
    DatasetSpec(
        name="Mouse_Brain_E11_MISAR",
        rna_filename="adata_RNA.h5ad",
        secondary_filename="adata_ATAC.h5ad",
        secondary_type="atac",
        platform="Spatial-epigenome-transcriptome",
        epochs=600,
        spatial_neighbors=5,
        feature_neighbors=30,
        cluster_pca_dims=15,
        pretrain_ratio=0.2,
        labels=LabelSpec(
            kind="csv",
            filename="anno.csv",
            barcode_column="barcode",
            label_column="cluster",
        ),
    ),
    DatasetSpec(
        name="Mouse_Brain_E13_MISAR",
        rna_filename="adata_RNA.h5ad",
        secondary_filename="adata_ATAC.h5ad",
        secondary_type="atac",
        platform="Spatial-epigenome-transcriptome",
        epochs=1500,
        spatial_neighbors=5,
        feature_neighbors=20,
        cluster_pca_dims=15,
        pretrain_ratio=0.1,
        labels=LabelSpec(
            kind="csv",
            filename="anno.csv",
            barcode_column="barcode",
            label_column="cluster",
        ),
    ),
    DatasetSpec(
        name="Mouse_Brain_E15_MISAR",
        rna_filename="adata_RNA.h5ad",
        secondary_filename="adata_ATAC.h5ad",
        secondary_type="atac",
        platform="Spatial-epigenome-transcriptome",
        epochs=1200,
        spatial_neighbors=4,
        feature_neighbors=20,
        cluster_pca_dims=10,
        pretrain_ratio=0.2,
        labels=LabelSpec(
            kind="csv",
            filename="anno.csv",
            barcode_column="barcode",
            label_column="cluster",
        ),
    ),
    DatasetSpec(
        name="Mouse_Brain_E18_MISAR",
        rna_filename="adata_RNA.h5ad",
        secondary_filename="adata_ATAC.h5ad",
        secondary_type="atac",
        platform="Spatial-epigenome-transcriptome",
        epochs=1600,
        spatial_neighbors=6,
        feature_neighbors=30,
        cluster_pca_dims=25,
        pretrain_ratio=0.0,
        labels=LabelSpec(
            kind="csv",
            filename="anno.csv",
            barcode_column="barcode",
            label_column="cluster",
        ),
    ),
    DatasetSpec(
        name="Mouse_Spleen",
        rna_filename="adata_RNA.h5ad",
        secondary_filename="adata_ADT.h5ad",
        secondary_type="adt",
        platform="SPOTS",
        epochs=220,
        spatial_neighbors=4,
        feature_neighbors=10,
        cluster_pca_dims=30,
        pretrain_ratio=0.3,
        labels=LabelSpec(
            kind="csv",
            filename="annotation_spleen.csv",
            barcode_column="Barcode",
            label_column="manual",
        ),
    ),
    DatasetSpec(
        name="Mouse_Spleen1",
        rna_filename="adata_RNA.h5ad",
        secondary_filename="adata_ADT.h5ad",
        secondary_type="adt",
        platform="SPOTS",
        epochs=400,
        spatial_neighbors=3,
        feature_neighbors=30,
        cluster_pca_dims=30,
        pretrain_ratio=0.2,
        labels=LabelSpec(
            kind="csv",
            filename="annotation_spleen1.csv",
            barcode_column="Barcode",
            label_column="manual",
        ),
    ),
    DatasetSpec(
        name="Mouse_Spleen2",
        rna_filename="adata_RNA.h5ad",
        secondary_filename="adata_ADT.h5ad",
        secondary_type="adt",
        platform="SPOTS",
        epochs=1200,
        spatial_neighbors=5,
        feature_neighbors=30,
        cluster_pca_dims=15,
        pretrain_ratio=0.1,
        labels=LabelSpec(
            kind="csv",
            filename="annotation_spleen2.csv",
            barcode_column="Barcode",
            label_column="manual",
        ),
    ),
    DatasetSpec(
        name="Simulation",
        rna_filename="adata_RNA.h5ad",
        secondary_filename="adata_ADT.h5ad",
        secondary_type="adt",
        platform="SPOTS",
        epochs=200,
        spatial_neighbors=4,
        feature_neighbors=10,
        cluster_pca_dims=20,
        pretrain_ratio=0.0,
        labels=LabelSpec(
            kind="text",
            filename="GT.txt",
            numeric_barcode_order=True,
        ),
    ),
)

DATASET_BY_NAME: Dict[str, DatasetSpec] = {item.name: item for item in DATASETS}


def select_datasets(names: Optional[Iterable[str]] = None) -> Tuple[DatasetSpec, ...]:
    if names is None:
        return DATASETS

    selected = []
    seen = set()
    lowercase_names = {name.lower(): name for name in DATASET_BY_NAME}
    for raw_name in names:
        canonical_name = lowercase_names.get(str(raw_name).strip().lower())
        if canonical_name is None:
            available = ", ".join(DATASET_BY_NAME)
            raise ValueError(f"Unknown dataset '{raw_name}'. Available datasets: {available}")
        if canonical_name not in seen:
            selected.append(DATASET_BY_NAME[canonical_name])
            seen.add(canonical_name)
    return tuple(selected)
