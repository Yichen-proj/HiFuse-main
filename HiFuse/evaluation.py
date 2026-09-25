import os
import subprocess
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    fowlkes_mallows_score,
    mutual_info_score,
    normalized_mutual_info_score,
    v_measure_score,
)

from .preprocess import pca


def pairwise_f1_jaccard(true_labels, pred_labels):
    contingency = pd.crosstab(
        pd.Series(pred_labels).astype(str),
        pd.Series(true_labels).astype(str),
    )
    if contingency.empty:
        return 0.0, 0.0

    counts = contingency.to_numpy(dtype=np.float64)
    choose_two = lambda values: values * (values - 1.0) / 2.0
    true_positive = float(choose_two(counts).sum())
    false_positive = float(choose_two(counts.sum(axis=1)).sum() - true_positive)
    false_negative = float(choose_two(counts.sum(axis=0)).sum() - true_positive)

    f1_denominator = 2.0 * true_positive + false_positive + false_negative
    jaccard_denominator = true_positive + false_positive + false_negative
    f1 = 0.0 if f1_denominator == 0.0 else 2.0 * true_positive / f1_denominator
    jaccard = 0.0 if jaccard_denominator == 0.0 else true_positive / jaccard_denominator
    return f1, jaccard


def compute_external_metrics(true_labels, pred_labels):
    truth = pd.Series(true_labels).astype(str)
    prediction = pd.Series(pred_labels).astype(str)
    pairwise_f1, pairwise_jaccard = pairwise_f1_jaccard(truth, prediction)
    return {
        "MI": float(mutual_info_score(truth, prediction)),
        "NMI": float(normalized_mutual_info_score(truth, prediction)),
        "AMI": float(adjusted_mutual_info_score(truth, prediction)),
        "FMI": float(fowlkes_mallows_score(truth, prediction)),
        "ARI": float(adjusted_rand_score(truth, prediction)),
        "VMeasure": float(v_measure_score(truth, prediction)),
        "F1": float(pairwise_f1),
        "Jaccard": float(pairwise_jaccard),
        "Completeness": float(completeness_score(truth, prediction)),
    }


def mclust_R(
    adata,
    num_cluster,
    modelNames="EEE",
    used_obsm="emb_pca",
    random_seed=2020,
):
    np.random.seed(random_seed)
    embedding = np.asarray(adata.obsm[used_obsm], dtype=float)

    with tempfile.TemporaryDirectory() as temporary_directory:
        input_csv = os.path.join(temporary_directory, "mclust_input.csv")
        output_csv = os.path.join(temporary_directory, "mclust_output.csv")
        pd.DataFrame(embedding).to_csv(input_csv, index=False)

        r_code = """
        args <- commandArgs(trailingOnly = TRUE)
        input_csv <- args[1]
        output_csv <- args[2]
        num_cluster <- as.integer(args[3])
        model_name <- args[4]
        random_seed <- as.integer(args[5])

        suppressPackageStartupMessages(library(mclust))
        set.seed(random_seed)

        x <- as.matrix(read.csv(input_csv, header = TRUE, check.names = FALSE))
        res <- suppressWarnings(Mclust(x, G = num_cluster, modelNames = model_name))
        if (is.null(res$classification) || length(res$classification) == 0) {
            message("mclust model ", model_name, " returned no classification; retrying with automatic model selection.")
            res <- suppressWarnings(Mclust(x, G = num_cluster))
        }
        if (is.null(res$classification) || length(res$classification) == 0) {
            stop("mclust did not return classifications")
        }
        write.csv(data.frame(cluster = res$classification), output_csv, row.names = FALSE, quote = FALSE)
        """
        completed = subprocess.run(
            [
                "Rscript",
                "-e",
                r_code,
                input_csv,
                output_csv,
                str(num_cluster),
                modelNames,
                str(random_seed),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if completed.stderr:
            print(completed.stderr)
        if not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0:
            raise RuntimeError("mclust produced an empty output file")
        clusters = pd.read_csv(output_csv)["cluster"].to_numpy()

    adata.obs["mclust"] = pd.Categorical(clusters.astype(int))
    return adata


def _resolve_clustering_representation(adata, key, use_pca, n_comps):
    if not use_pca:
        return key
    representation_key = f"{key}_pca"
    adata.obsm[representation_key] = pca(
        adata,
        use_reps=key,
        n_comps=n_comps,
    )
    return representation_key


def clustering(
    adata,
    n_clusters=7,
    key="emb",
    add_key="HiFuse",
    method="mclust",
    start=0.1,
    end=3.0,
    increment=0.01,
    use_pca=False,
    n_comps=20,
    mclust_model="EEE",
):
    representation = _resolve_clustering_representation(
        adata,
        key,
        use_pca,
        n_comps,
    )
    if method == "mclust":
        mclust_R(
            adata,
            used_obsm=representation,
            num_cluster=n_clusters,
            modelNames=mclust_model,
        )
        adata.obs[add_key] = adata.obs["mclust"]
        return

    if method not in {"leiden", "louvain"}:
        raise ValueError(f"Unsupported clustering method: {method}")
    resolution = search_res(
        adata,
        n_clusters,
        method=method,
        use_rep=representation,
        start=start,
        end=end,
        increment=increment,
    )
    cluster = sc.tl.leiden if method == "leiden" else sc.tl.louvain
    cluster(adata, random_state=0, resolution=resolution)
    adata.obs[add_key] = adata.obs[method]


def search_res(
    adata,
    n_clusters,
    method="leiden",
    use_rep="emb",
    start=0.1,
    end=3.0,
    increment=0.01,
):
    if method not in {"leiden", "louvain"}:
        raise ValueError(f"Unsupported clustering method: {method}")

    print("Searching resolution...")
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    cluster = sc.tl.leiden if method == "leiden" else sc.tl.louvain
    for resolution in sorted(np.arange(start, end, increment), reverse=True):
        cluster(adata, random_state=0, resolution=resolution)
        count = adata.obs[method].nunique()
        print(f"resolution={resolution}, cluster number={count}")
        if count == n_clusters:
            return resolution
    raise AssertionError(
        "Resolution is not found. Please try a bigger range or a smaller step."
    )


def plot_weight_value(alpha, label, modality1="mRNA", modality2="protein"):
    values = pd.DataFrame(
        {
            modality1: alpha[:, 0],
            modality2: alpha[:, 1],
            "label": label,
        }
    )
    values = values.set_index("label").stack().reset_index()
    values.columns = ["HiFuse label", "Modality", "Weight value"]
    axis = sns.violinplot(
        data=values,
        x="HiFuse label",
        y="Weight value",
        hue="Modality",
        split=True,
        inner="quart",
        linewidth=1,
        show=False,
    )
    axis.set_title(f"{modality1} vs {modality2}")
    plt.tight_layout(w_pad=0.05)
    plt.show()
