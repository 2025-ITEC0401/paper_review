# -*- coding: utf-8 -*-
"""
학습 데이터 생성 관련 함수들을 관리하는 모듈
"""
import numpy as np
from .config import (
    N_SAMPLES, SEQ_LENGTH, N_FEATURES,
    NOISE_LEVEL_RANGE,
    SIMPLE_SINE_FREQ_RANGE, SIMPLE_SINE_AMP_RANGE,
    TREND_SINE_FREQ_RANGE, TREND_SINE_AMP_RANGE, TREND_SLOPE_RANGE,
    COMPLEX_FREQ1_RANGE, COMPLEX_FREQ2_RANGE, COMPLEX_AMP1_RANGE, COMPLEX_AMP2_RANGE
)

def generate_training_data():
    """새로운 학습 데이터를 생성하는 함수"""
    print("새로운 학습 데이터를 생성합니다...")
    
    n_samples = N_SAMPLES
    seq_length = SEQ_LENGTH
    n_features = N_FEATURES
    
    # 더욱 명확한 패턴의 시계열 데이터 생성 (Loss 최적화를 위해)
    x = np.linspace(0, 4*np.pi, seq_length)  # 적절한 시간 범위
    test_data = np.zeros((n_samples, seq_length, n_features))
    pattern_labels = []  # 패턴 라벨 저장
    
    for i in range(n_samples):
        # 더 구분되는 패턴 생성 (균등 분배)
        pattern_type = i % 3  # 패턴을 순서대로 반복하여 균형 맞춤
        pattern_labels.append(pattern_type)
        
        if pattern_type == 0:
            # 저주파 단순 사인파 (명확히 구분)
            freq = np.random.uniform(*SIMPLE_SINE_FREQ_RANGE)
            phase = np.random.uniform(0, np.pi/2)
            amplitude = np.random.uniform(*SIMPLE_SINE_AMP_RANGE)
            signal = amplitude * np.sin(freq * x + phase)
        elif pattern_type == 1:
            # 선형 트렌드가 강한 사인파
            freq = np.random.uniform(*TREND_SINE_FREQ_RANGE)
            phase = np.random.uniform(0, np.pi/2) 
            amplitude = np.random.uniform(*TREND_SINE_AMP_RANGE)
            trend_slope = np.random.uniform(*TREND_SLOPE_RANGE)
            trend = trend_slope * x
            signal = amplitude * np.sin(freq * x + phase) + trend
        else:
            # 고주파 복합파 (명확히 다른 주파수)
            freq1 = np.random.uniform(*COMPLEX_FREQ1_RANGE)
            freq2 = np.random.uniform(*COMPLEX_FREQ2_RANGE)
            phase1 = np.random.uniform(0, np.pi/2)
            phase2 = np.random.uniform(0, np.pi/2)
            amp1 = np.random.uniform(*COMPLEX_AMP1_RANGE)
            amp2 = np.random.uniform(*COMPLEX_AMP2_RANGE)
            signal = amp1 * np.sin(freq1 * x + phase1) + amp2 * np.sin(freq2 * x + phase2)
        
        # 노이즈 감소 (패턴을 더 명확하게)
        noise_level = np.random.uniform(*NOISE_LEVEL_RANGE)
        noise = np.random.normal(0, noise_level, seq_length)
        
        test_data[i, :, 0] = signal + noise
    
    print("테스트 데이터 생성 완료: " + str(test_data.shape))
    
    # 데이터 정규화 (Z-score normalization) - Loss 최적화를 위해
    print("데이터 정규화 중...")
    for i in range(test_data.shape[0]):
        # 각 시계열을 개별적으로 정규화
        series = test_data[i, :, 0]
        mean_val = np.mean(series)
        std_val = np.std(series)
        if std_val > 0:  # 0으로 나누기 방지
            test_data[i, :, 0] = (series - mean_val) / std_val
    
    print("데이터 정규화 완료!")
    
    return test_data, np.array(pattern_labels)

def create_labeled_dataset(data, pattern_types):
    """라벨이 있는 데이터셋 생성 (분류 테스크용)"""
    labels = []
    for i in range(len(data)):
        # 데이터 생성과 동일한 패턴 라벨링 방식 사용
        pattern_type = i % 3  # 0: 단순 사인파, 1: 트렌드 사인파, 2: 복합 주파수
        labels.append(pattern_type)
    
    return np.array(labels)
