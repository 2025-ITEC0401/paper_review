# -*- coding: utf-8 -*-
"""
설정 변수들을 관리하는 모듈
"""

# ===============================
# 학습 관련 설정 변수
# ===============================
# 데이터 생성 설정
N_SAMPLES = 1000           # 시계열 개수
SEQ_LENGTH = 500           # 시계열 길이
N_FEATURES = 1             # 특성 개수 (univariate)

# 모델 하이퍼파라미터
OUTPUT_DIMS = 512     # 더 풍부한 표현
HIDDEN_DIMS = 256     # 비례적으로 증가
DEPTH = 10                 # 네트워크 깊이
LEARNING_RATE = 0.0001     # 학습률
BATCH_SIZE = 64            # 배치 크기

# 학습 설정
N_EPOCHS = 10             # 에포크 수
ADDITIONAL_EPOCHS = 10    # 추가 학습 에포크 수
ENABLE_ADDITIONAL_TRAINING = False  # 추가 학습 활성화 여부
MODEL_PATH = "model/ts2vec_model.pt"  # 모델 저장 경로

# 데이터 저장/로드 설정
SAVE_TRAINING_DATA = True  # 학습 데이터 저장 여부
DATA_PATH = "result/training_data.npz"  # 데이터 저장 경로
USE_SAVED_DATA = False      # 저장된 데이터 사용 여부

# 데이터 생성 패턴 설정
NOISE_LEVEL_RANGE = (0.02, 0.08)  # 노이즈 레벨 범위

# 패턴별 파라미터
SIMPLE_SINE_FREQ_RANGE = (0.5, 1.2)
SIMPLE_SINE_AMP_RANGE = (1.5, 2.5)

TREND_SINE_FREQ_RANGE = (1.0, 1.8)
TREND_SINE_AMP_RANGE = (0.8, 1.5)
TREND_SLOPE_RANGE = (0.002, 0.005)

COMPLEX_FREQ1_RANGE = (2.5, 3.5)
COMPLEX_FREQ2_RANGE = (4.0, 5.0)
COMPLEX_AMP1_RANGE = (1.0, 1.5)
COMPLEX_AMP2_RANGE = (0.5, 0.8)

# 시각화 설정
N_SAMPLE_PLOTS = 5         # 시각화할 샘플 개수
TOP_K_SIMILARITY = 5       # 유사성 검색 상위 K개
