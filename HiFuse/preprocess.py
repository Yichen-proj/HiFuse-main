import os
import random
from typing import Optional

import anndata
import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
import sklearn
import torch
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from torch.backends import cudnn


_SPATIAL_NEIGHBOR_DEFAULTS = {
    "Stereo-CITE-seq": 6,
    "Spatial-epigenome-transcriptome": 6,
}


def _resolve_spatial_neighbors(datatype, n_neighbors):
    if n_neighbors is not None:
        return n_neighbors
    return _SPATIAL_NEIGHBOR_DEFAULTS.get(datatype, 3)


def construct_neighbor_graph(
    adata_omics1,
    adata_omics2,
    datatype="SPOTS",
    n_neighbors=3,
    feature_k=20,
):
    spatial_k = _resolve_spatial_neighbors(datatype, n_neighbors)
    modalities = (adata_omics1, adata_omics2)

    for adata in modalities:
        adata.uns["adj_spatial"] = construct_graph_by_coordinate(
            adata.obsm["spatial"],
            n_neighbors=spatial_k,
        )

    feature_graphs = construct_graph_by_feature(*modalities, k=feature_k)
    for adata, graph in zip(modalities, feature_graphs):
        adata.obsm["adj_feature"] = graph

    return {
        "adata_omics1": adata_omics1,
        "adata_omics2": adata_omics2,
    }


def pca(adata, use_reps=None, n_comps=10):
    from scipy.sparse.csc import csc_matrix
    from scipy.sparse.csr import csr_matrix
    from sklearn.decomposition import PCA

    values = adata.obsm[use_reps] if use_reps is not None else adata.X
    if isinstance(values, (csc_matrix, csr_matrix)):
        values = values.toarray()
    return PCA(n_components=n_comps).fit_transform(values)


def clr_normalize_each_cell(adata, inplace=True):
    def seurat_clr(row):
        log_sum = np.sum(np.log1p(row[row > 0]))
        scale = np.exp(log_sum / len(row))
        return np.log1p(row / scale)

    target = adata if inplace else adata.copy()
    matrix = (
        target.X.toarray() if scipy.sparse.issparse(target.X) else np.asarray(target.X)
    )
    target.X = np.apply_along_axis(seurat_clr, 1, matrix)
    return target


def construct_graph_by_feature(
    adata_omics1,
    adata_omics2,
    k=20,
    mode="connectivity",
    metric="correlation",
    include_self=False,
):
    options = {
        "n_neighbors": k,
        "mode": mode,
        "metric": metric,
        "include_self": include_self,
    }
    return tuple(
        kneighbors_graph(adata.obsm["feat"], **options)
        for adata in (adata_omics1, adata_omics2)
    )


def construct_graph_by_coordinate(cell_position, n_neighbors=3):
    coordinates = np.asarray(cell_position, dtype=np.float32)
    neighbor_model = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coordinates)
    _, indices = neighbor_model.kneighbors(coordinates)
    sources = indices[:, 0].repeat(n_neighbors)
    targets = indices[:, 1:].flatten()
    return pd.DataFrame(
        {
            "x": sources,
            "y": targets,
            "value": np.ones(sources.size),
        }
    )


def transform_adjacent_matrix(adjacent):
    n_spots = adjacent["x"].max() + 1
    return coo_matrix(
        (adjacent["value"], (adjacent["x"], adjacent["y"])),
        shape=(n_spots, n_spots),
    )


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    return torch.sparse.FloatTensor(indices, values, torch.Size(sparse_mx.shape))


def preprocess_graph(adj):
    adjacency = sp.coo_matrix(adj)
    with_self_loops = adjacency + sp.eye(adjacency.shape[0])
    row_sum = np.array(with_self_loops.sum(1))
    inv_sqrt_degree = sp.diags(np.power(row_sum, -0.5).flatten())
    normalized = (
        with_self_loops.dot(inv_sqrt_degree).transpose().dot(inv_sqrt_degree).tocoo()
    )
    return sparse_mx_to_torch_sparse_tensor(normalized)


def _symmetrize_binary_adjacency(adj):
    symmetric = adj + adj.T
    return np.where(symmetric > 1, 1, symmetric)


def _prepare_spatial_adjacency(adata):
    adjacency = transform_adjacent_matrix(adata.uns["adj_spatial"]).toarray()
    return preprocess_graph(_symmetrize_binary_adjacency(adjacency))


def _prepare_feature_adjacency(adata):
    adjacency = torch.FloatTensor(adata.obsm["adj_feature"].copy().toarray())
    return preprocess_graph(_symmetrize_binary_adjacency(adjacency))


def adjacent_matrix_preprocessing(adata_omics1, adata_omics2):
    spatial1 = _prepare_spatial_adjacency(adata_omics1)
    spatial2 = _prepare_spatial_adjacency(adata_omics2)
    feature1 = _prepare_feature_adjacency(adata_omics1)
    feature2 = _prepare_feature_adjacency(adata_omics2)
    return {
        "adj_spatial_omics1": spatial1,
        "adj_spatial_omics2": spatial2,
        "adj_feature_omics1": feature1,
        "adj_feature_omics2": feature2,
    }


def lsi(
    adata: anndata.AnnData,
    n_components: int = 20,
    use_highly_variable: Optional[bool] = None,
    **kwargs,
) -> None:
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    transformed = tfidf(adata_use.X)
    normalized = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(transformed)
    normalized = np.log1p(normalized * 1e4)
    embedding = sklearn.utils.extmath.randomized_svd(
        normalized,
        n_components,
        **kwargs,
    )[0]
    embedding -= embedding.mean(axis=1, keepdims=True)
    standard_deviation = embedding.std(axis=1, ddof=1, keepdims=True)
    standard_deviation[standard_deviation == 0] = 1
    embedding /= standard_deviation
    adata.obsm["X_lsi"] = embedding[:, 1:]


def tfidf(X):
    column_sum = np.asarray(X.sum(axis=0)).reshape(-1)
    column_sum = np.where(column_sum == 0, 1, column_sum)
    inverse_document_frequency = X.shape[0] / column_sum

    if scipy.sparse.issparse(X):
        row_sum = np.asarray(X.sum(axis=1)).reshape(-1)
        row_sum = np.where(row_sum == 0, 1, row_sum)
        term_frequency = X.multiply(1 / row_sum[:, None])
        return term_frequency.multiply(inverse_document_frequency)

    row_sum = np.asarray(X.sum(axis=1)).reshape(-1, 1)
    row_sum = np.where(row_sum == 0, 1, row_sum)
    return (X / row_sum) * inverse_document_frequency


def fix_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
