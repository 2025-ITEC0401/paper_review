# -*- coding: utf-8 -*-
"""
데이터 저장/로드 관련 함수들을 관리하는 모듈
"""
import numpy as np
import os
from .config import DATA_PATH

def save_training_data(data, labels, filepath=DATA_PATH):
    """학습 데이터를 저장하는 함수"""
    try:
        np.savez_compressed(filepath, data=data, labels=labels)
        print(f"학습 데이터를 {filepath}에 저장했습니다.")
        print(f"데이터 크기: {data.shape}, 라벨 크기: {labels.shape}")
        return True
    except Exception as e:
        print(f"데이터 저장 실패: {e}")
        return False

def load_training_data(filepath=DATA_PATH):
    """저장된 학습 데이터를 로드하는 함수"""
    try:
        if not os.path.exists(filepath):
            print(f"저장된 데이터 파일이 없습니다: {filepath}")
            return None, None
        
        loaded = np.load(filepath)
        data = loaded['data']
        labels = loaded['labels']
        print(f"저장된 학습 데이터를 {filepath}에서 로드했습니다.")
        print(f"데이터 크기: {data.shape}, 라벨 크기: {labels.shape}")
        return data, labels
    except Exception as e:
        print(f"데이터 로드 실패: {e}")
        return None, None
