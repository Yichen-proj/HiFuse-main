# HiFuse

HiFuse is a spatial multi-omics integration method that learns a joint representation from paired molecular modalities and spatial information. The repository contains the core implementation and a single reproducible runner for the 14 supported datasets.

![HiFuse workflow](Framework.png)

## Overview

HiFuse accepts paired modalities measured at shared tissue locations and learns a unified representation for spatial domain identification. First, each modality is preprocessed, and a modality-specific feature graph is paired with one shared spatial graph. Next, a dual-graph encoder extracts molecular-similarity and spatial-proximity representations. HAF then integrates graph views, modalities, and complementary fusion paths through three attention levels. Finally, SCCO aligns paired spots while preserving input and modality-specific information.

## Requirements

- `python==3.8`
- `torch>=1.8.0`
- `cudnn>=10.2`
- `pandas==1.4.2`
- `matplotlib==3.4.2`
- `anndata==0.8.0`
- `scikit-learn==1.1.1`
- `rpy2==3.4.1`
- `numpy==1.22.3`
- `tqdm==4.64.0`
- `scipy==1.8.1`
- `R==4.0.3`
- `scanpy==1.9.1`
- `scikit-misc==0.2.0`

The clustering step uses the R package `mclust`; therefore, `Rscript` and `mclust` must be available in the activated environment.

## Data

HiFuse was evaluated on 14 publicly available spatial multi-omics datasets, including four from the 10x Genomics Visium platform, three from the SPOTS platform, four from the MISAR-seq platform, two from the spatial epigenome-transcriptome platform, and one simulated dataset. All public datasets used by HiFuse can be downloaded free of charge from Zenodo: [Spatial multi-omics datasets for HiFuse](https://doi.org/10.5281/zenodo.22986116).

## Run

From the project root, run all supported datasets with:

```bash
python scripts/run_14_datasets.py
```

The runner trains HiFuse with the fixed dataset-specific baseline settings and writes each dataset's embedding, predictions, metrics and comparison plot to `results/baseline_14/<dataset>/`. Combined CSV and JSON summaries are written under `results/baseline_14/`.
