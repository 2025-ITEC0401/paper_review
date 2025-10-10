# -*- coding: utf-8 -*-
"""
TS2Vec 모델 테스트 메인 스크립트

기능별로 분리된 모듈들을 import하여 전체 파이프라인을 실행합니다.
"""
import numpy as np
import torch
import os
from core import TS2Vec

# 분리된 모듈들을 import
from src import (
    # Config
    N_SAMPLES, SEQ_LENGTH, N_FEATURES, USE_SAVED_DATA, SAVE_TRAINING_DATA, N_SAMPLE_PLOTS,
    
    # Functions
    generate_training_data, load_training_data, save_training_data,
    plot_sample_data, create_model, train_or_load_model, run_downstream_tasks
)


def test_ts2vec():
    """TS2Vec 모델을 테스트하는 메인 함수"""
    print("TS2Vec 테스트를 시작합니다...")
    
    # 데이터 로드/생성 결정
    test_data = None
    pattern_labels = None
    
    if USE_SAVED_DATA:
        print("저장된 학습 데이터 사용을 시도합니다...")
        test_data, pattern_labels = load_training_data()
    
    if test_data is None:
        print("저장된 데이터가 없거나 로드에 실패했습니다.")
        test_data, pattern_labels = generate_training_data()
        
        # 새로 생성한 데이터 저장
        if SAVE_TRAINING_DATA:
            save_training_data(test_data, pattern_labels)
    else:
        print("저장된 학습 데이터를 성공적으로 로드했습니다!")
        print("동일한 데이터로 일관성 있는 실험이 가능합니다.")
    
    # 데이터 차원 정보 설정
    n_samples = test_data.shape[0]
    seq_length = test_data.shape[1] 
    n_features = test_data.shape[2]
    
    # 입력 데이터 시각화
    print("입력 데이터 시각화 중...")
    plot_sample_data(test_data, n_samples=N_SAMPLE_PLOTS, save_path="result/input_data_samples.png")
    
    # 디바이스 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("사용하는 디바이스: " + device)
    
    # 모델 생성
    model = create_model(n_features, device)
    
    # 모델 학습 또는 로드
    model = train_or_load_model(model, test_data)
    
    # 표현 학습 테스트
    print("표현 학습 테스트...")
    try:
        representations = model.encode(test_data)
        print("학습된 표현 크기: " + str(representations.shape))
        
        # 간단한 성능 확인 - 표현 벡터의 통계
        print("표현 벡터 통계:")
        print("  - 평균: " + str(np.mean(representations)))
        print("  - 표준편차: " + str(np.std(representations)))
        print("  - 최솟값: " + str(np.min(representations)))
        print("  - 최댓값: " + str(np.max(representations)))
        
        # 다운스트림 테스크 실행
        downstream_results = run_downstream_tasks(model, test_data)
        
        print("\n" + "="*60)
        print("TS2Vec 전체 테스트가 성공적으로 완료되었습니다!")
        print("="*60)
        
    except Exception as e:
        print("테스트 중 오류 발생: " + str(e))
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_ts2vec()