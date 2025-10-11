# -*- coding: utf-8 -*-
"""
모델 학습 관련 함수들을 관리하는 모듈
"""
import os
import torch
from core import TS2Vec
from .config import (
    OUTPUT_DIMS, HIDDEN_DIMS, DEPTH, LEARNING_RATE, BATCH_SIZE,
    N_EPOCHS, ADDITIONAL_EPOCHS, ENABLE_ADDITIONAL_TRAINING, MODEL_PATH
)
from .visualization import plot_loss_curve

def create_model(n_features, device='cpu'):
    """TS2Vec 모델을 생성하고 초기화하는 함수"""
    model = TS2Vec(
        input_dims=n_features,
        output_dims=OUTPUT_DIMS,
        hidden_dims=HIDDEN_DIMS,
        depth=DEPTH,
        device=device,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        max_train_length=None
    )
    print("TS2Vec 모델 초기화 완료")
    return model

def train_or_load_model(model, test_data, model_path=MODEL_PATH):
    """모델을 학습하거나 기존 모델을 로드하는 함수"""
    # 기존 모델이 있는지 확인
    if os.path.exists(model_path):
        print("기존 모델을 발견했습니다. 모델을 로드합니다...")
        try:
            model.load(model_path)
            print("모델 로드 완료!")
            
            # 추가 학습 수행 여부 확인
            if ENABLE_ADDITIONAL_TRAINING and ADDITIONAL_EPOCHS > 0:
                print(f"\n기존 모델을 추가로 {ADDITIONAL_EPOCHS} 에포크 학습합니다...")
                print("이전 학습을 이어서 진행합니다 (Fine-tuning)")
                
                # 추가 학습 실행
                additional_loss_log = model.fit(
                    train_data=test_data,
                    n_epochs=ADDITIONAL_EPOCHS,
                    verbose=True
                )
                print("추가 학습 완료! 손실 로그: " + str(additional_loss_log[-5:]))
                
                # 추가 학습 손실 그래프 생성
                plot_loss_curve(additional_loss_log, save_path="result/additional_training_loss_curve.png")
                
                # 업데이트된 모델 저장
                model.save(model_path)
                print("추가 학습된 모델을 " + model_path + "에 저장했습니다.")
            else:
                print("추가 학습이 비활성화되어 기존 모델을 그대로 사용합니다.")
                
        except Exception as e:
            print("모델 로드 실패: " + str(e))
            print("새로운 모델을 훈련합니다...")
            _train_new_model(model, test_data, model_path)
    else:
        print("기존 모델이 없습니다. 새로운 모델을 훈련합니다...")
        _train_new_model(model, test_data, model_path)
    
    return model

def _train_new_model(model, test_data, model_path):
    """새로운 모델을 훈련하는 내부 함수"""
    try:
        loss_log = model.fit(
            train_data=test_data,
            n_epochs=N_EPOCHS,
            verbose=True
        )
        print("훈련 완료! 손실 로그: " + str(loss_log[-5:]))
        # 손실률 그래프 생성
        plot_loss_curve(loss_log, save_path="result/training_loss_curve.png")
        # 모델 저장
        model.save(model_path)
        print("모델을 " + model_path + "에 저장했습니다.")
    except Exception as e:
        print("훈련 중 오류 발생: " + str(e))
        import traceback
        traceback.print_exc()
        raise e
