# src/downstream/__init__.py

from .heads import (
    ClassificationHead,
    ClusteringHead,
    AnomalyDetectionHead,
    ForecastingHead
)
from .evaluate import evaluate_all_tasks, DATASET_TASKS, DATASETS
from .visualize import (
    visualize_representations,
    visualize_clustering,
    create_summary_figure
)

__all__ = [
    'ClassificationHead',
    'ClusteringHead',
    'AnomalyDetectionHead',
    'ForecastingHead',
    'evaluate_all_tasks',
    'DATASET_TASKS',
    'DATASETS',
    'visualize_representations',
    'visualize_clustering',
    'create_summary_figure'
]
