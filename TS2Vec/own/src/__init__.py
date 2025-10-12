# -*- coding: utf-8 -*-
"""
TS2Vec 프로젝트의 모듈들을 관리하는 패키지
"""

from .config import *
from .data_generator import generate_training_data, create_labeled_dataset
from .data_manager import save_training_data, load_training_data
from .visualization import plot_sample_data, plot_loss_curve
from .downstream_tasks import run_downstream_tasks
from .model_trainer import create_model, train_or_load_model

__all__ = [
    # Config
    'N_SAMPLES', 'SEQ_LENGTH', 'N_FEATURES', 'OUTPUT_DIMS', 'HIDDEN_DIMS', 'DEPTH',
    'LEARNING_RATE', 'BATCH_SIZE', 'N_EPOCHS', 'ADDITIONAL_EPOCHS', 'ENABLE_ADDITIONAL_TRAINING',
    'MODEL_PATH', 'SAVE_TRAINING_DATA', 'DATA_PATH', 'USE_SAVED_DATA', 'N_SAMPLE_PLOTS',
    'TOP_K_SIMILARITY',
    
    # Data functions
    'generate_training_data', 'create_labeled_dataset', 'save_training_data', 'load_training_data',
    
    # Visualization
    'plot_sample_data', 'plot_loss_curve',
    
    # Downstream tasks
    'run_downstream_tasks',
    
    # Model training
    'create_model', 'train_or_load_model'
]
