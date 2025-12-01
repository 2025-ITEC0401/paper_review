# src/config.py
# Merit 학습 설정

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainingConfig:
    """학습 관련 설정"""
    # 기본 설정
    dataset_name: str = 'BasicMotions'
    root_path: str = './datasets/'
    llm_path: str = './models/llama-3.1-8b-instruct'
    gpu: int = 0

    # 시퀀스 및 표현 설정
    seq_len: int = 96
    repr_dim: int = 320

    # 학습 하이퍼파라미터 (논문 기반)
    epochs: int = 100
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 5e-4

    # Memory Bank 설정
    bank_size: int = 100
    k_candidates: int = 5  # 초기 후보 수
    m_relevant: int = 3    # 선택할 관련 시퀀스 수

    # Contrastive Learning 설정
    temperature: float = 0.1

    # 스케줄러 설정
    use_scheduler: bool = False
    warmup_ratio: float = 0.1

    # Early Stopping
    patience: int = 5


@dataclass
class DatasetConfig:
    """데이터셋별 설정"""
    name: str
    channels: int
    length: int
    num_classes: int
    tasks: List[str] = field(default_factory=list)


# 지원 데이터셋 정의
DATASET_CONFIGS = {
    'AtrialFibrillation': DatasetConfig(
        name='AtrialFibrillation',
        channels=2,
        length=640,
        num_classes=3,
        tasks=['classification', 'anomaly_detection', 'forecasting']
    ),
    'StandWalkJump': DatasetConfig(
        name='StandWalkJump',
        channels=4,
        length=2500,
        num_classes=3,
        tasks=['classification', 'clustering', 'anomaly_detection']
    ),
    'ArticularyWordRecognition': DatasetConfig(
        name='ArticularyWordRecognition',
        channels=9,
        length=144,
        num_classes=25,
        tasks=['classification']
    ),
    'NATOPS': DatasetConfig(
        name='NATOPS',
        channels=24,
        length=51,
        num_classes=6,
        tasks=['classification']
    ),
    'PenDigits': DatasetConfig(
        name='PenDigits',
        channels=2,
        length=8,
        num_classes=10,
        tasks=['classification', 'clustering']
    ),
    'UWaveGestureLibrary': DatasetConfig(
        name='UWaveGestureLibrary',
        channels=3,
        length=315,
        num_classes=8,
        tasks=['classification']
    ),
    'PEMS-SF': DatasetConfig(
        name='PEMS-SF',
        channels=963,
        length=144,
        num_classes=7,
        tasks=['anomaly_detection', 'forecasting', 'imputation']
    ),
}


# 다운스트림 태스크 설정
DOWNSTREAM_TASKS = {
    'classification': {
        'methods': ['svm', 'knn', 'rf'],
        'metrics': ['accuracy', 'f1_macro', 'f1_weighted']
    },
    'clustering': {
        'methods': ['kmeans'],
        'metrics': ['nmi', 'ri', 'ari', 'silhouette']
    },
    'anomaly_detection': {
        'methods': ['isolation_forest', 'one_class_svm'],
        'metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    },
    'forecasting': {
        'methods': ['ridge'],
        'metrics': ['mse', 'rmse', 'mae', 'r2']
    },
    'imputation': {
        'methods': ['knn'],
        'metrics': ['mse', 'rmse', 'mae']
    }
}


def get_dataset_config(dataset_name: str) -> Optional[DatasetConfig]:
    """데이터셋 설정 반환"""
    return DATASET_CONFIGS.get(dataset_name)


def get_supported_datasets() -> List[str]:
    """지원 데이터셋 목록 반환"""
    return list(DATASET_CONFIGS.keys())
