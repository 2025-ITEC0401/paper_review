# src/downstream/evaluate.py
# 다운스트림 태스크 평가 모듈

import os
import numpy as np

from .heads import (
    ClassificationHead,
    ClusteringHead,
    AnomalyDetectionHead,
    ForecastingHead
)

# 데이터셋별 다운스트림 태스크 정의
DATASET_TASKS = {
    'AtrialFibrillation': ['classification', 'anomaly_detection', 'forecasting'],
    'StandWalkJump': ['classification', 'clustering', 'anomaly_detection'],
    'ArticularyWordRecognition': ['classification'],
    'NATOPS': ['classification'],
    'PenDigits': ['classification', 'clustering'],
    'UWaveGestureLibrary': ['classification'],
}

DATASETS = list(DATASET_TASKS.keys())


def load_representations(dataset_name, reps_dir):
    """저장된 representations와 labels 로드"""
    reps_path = os.path.join(reps_dir, f'{dataset_name}_representations.npy')
    labels_path = os.path.join(reps_dir, f'{dataset_name}_labels.npy')

    if not os.path.exists(reps_path):
        print(f"Warning: {reps_path} not found")
        return None, None

    representations = np.load(reps_path)
    labels = np.load(labels_path) if os.path.exists(labels_path) else None

    return representations, labels


def evaluate_all_tasks(dataset_name, representations, labels, tasks):
    """모든 다운스트림 태스크 평가"""
    results = {'dataset': dataset_name}

    if 'classification' in tasks:
        print(f"  [Classification] Evaluating...")
        clf_head = ClassificationHead(method='svm')
        results['classification'] = clf_head.evaluate(representations, labels)

        cv_results = clf_head.cross_validate(representations, labels)
        results['classification'].update(cv_results)

        for method in ['knn', 'rf']:
            clf_head_alt = ClassificationHead(method=method)
            alt_results = clf_head_alt.evaluate(representations, labels)
            results[f'classification_{method}'] = alt_results

    if 'clustering' in tasks:
        print(f"  [Clustering] Evaluating...")
        cluster_head = ClusteringHead()
        cluster_results, cluster_labels, centers = cluster_head.evaluate(representations, labels)
        results['clustering'] = cluster_results
        results['_cluster_labels'] = cluster_labels
        results['_cluster_centers'] = centers

    if 'anomaly_detection' in tasks:
        print(f"  [Anomaly Detection] Evaluating...")
        ad_head = AnomalyDetectionHead(method='isolation_forest')
        results['anomaly_detection_if'] = ad_head.evaluate(representations, labels)

        ad_head_svm = AnomalyDetectionHead(method='one_class_svm')
        results['anomaly_detection_ocsvm'] = ad_head_svm.evaluate(representations, labels)

    if 'forecasting' in tasks:
        print(f"  [Forecasting] Evaluating...")
        forecast_head = ForecastingHead(horizon=1)
        results['forecasting'] = forecast_head.evaluate(representations)

    return results
