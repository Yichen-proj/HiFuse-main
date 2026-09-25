"""HiFuse spatial multi-omics integration."""

from .datasets import DATASETS, DatasetSpec, select_datasets
from .evaluation import clustering, compute_external_metrics
from .model import HiFuseNetwork
from .pipeline import run_dataset
from .preprocess import construct_neighbor_graph, fix_seed
from .trainer import HiFuseTrainer

__all__ = [
    "DATASETS",
    "DatasetSpec",
    "HiFuseNetwork",
    "HiFuseTrainer",
    "clustering",
    "compute_external_metrics",
    "construct_neighbor_graph",
    "fix_seed",
    "run_dataset",
    "select_datasets",
]

__version__ = "0.1.0"
