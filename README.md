# HiFuse

HiFuse is a spatial multi-omics integration method that learns a joint representation from paired molecular modalities and spatial information. The repository contains the core implementation and a single reproducible runner for the 14 supported datasets.

![HiFuse workflow](Framework.png)

## Requirements

- `python==3.8`
- `torch>=1.8.0`
- `cudnn>=10.2`
- `numpy==1.22.3`
- `scanpy==1.9.1`
- `anndata==0.8.0`
- `rpy2==3.4.1`
- `pandas==1.4.2`
- `scipy==1.8.1`
- `scikit-learn==1.1.1`
- `scikit-misc==0.2.0`
- `tqdm==4.64.0`
- `matplotlib==3.4.2`
- `R==4.0.3`

The clustering step uses the R package `mclust`; therefore, `Rscript` and `mclust` must be available in the activated environment.

## Data

Place each dataset under `Data/<dataset>/`. The repository runner supports:

1. `HLN`
2. `Human_Brain_Hippocampal_Spatial-epigenome-transcriptome`
3. `Human_Lymph_Node_S2`
4. `Human_Lymph_Node_S3`
5. `Human_tonsil_s1_10x`
6. `Mouse_Brain`
7. `Mouse_Brain_E11_MISAR`
8. `Mouse_Brain_E13_MISAR`
9. `Mouse_Brain_E15_MISAR`
10. `Mouse_Brain_E18_MISAR`
11. `Mouse_Spleen`
12. `Mouse_Spleen1`
13. `Mouse_Spleen2`
14. `Simulation`

See `Data/README.md` for the expected directory layout.

## Run

From the project root, run all supported datasets with:

```bash
python scripts/run_14_datasets.py
```

The runner trains HiFuse with the fixed dataset-specific baseline settings and writes each dataset's embedding, predictions, metrics and comparison plot to `results/baseline_14/<dataset>/`. Combined CSV and JSON summaries are written under `results/baseline_14/`.

## Package layout

```text
HiFuse/       Core model, preprocessing, training and evaluation code
scripts/      Reproducible 14-dataset entry point
Data/         Input datasets
results/      Generated at runtime
```

## License

The project is distributed under the terms in `LICENSE.md`.
