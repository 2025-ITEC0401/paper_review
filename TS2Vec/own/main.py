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

        global_repr = np.mean(representations, axis=1)
    
    # 훈련/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        global_repr, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    # Random Forest 분류기
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    
    # 예측
    y_pred = rf_classifier.predict(X_test)
    
    # 정확도 계산
    accuracy = np.mean(y_pred == y_test)
    print("Classification Results:")
    print("  - Accuracy: {:.4f}".format(accuracy))
    print("  - Training samples: " + str(len(X_train)))
    print("  - Test samples: " + str(len(X_test)))
    
    # 혼동 행렬 시각화
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Simple Sine', 'Trend Sine', 'Complex Wave'],
                yticklabels=['Simple Sine', 'Trend Sine', 'Complex Wave'])
    plt.title('Time Series Classification Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print("Classification visualization saved to " + save_path)
    plt.close()
    
    return accuracy, y_pred, y_test

def downstream_similarity_search(representations, query_idx=0, top_k=5):
    """유사성 검색 다운스트림 테스크"""
    print("\n=== 시계열 유사성 검색 테스트 ===")
    
    # representations가 이미 (n_samples, output_dims) 형태인 경우
    if len(representations.shape) == 2:
        global_repr = representations
    else:
        # 전체 시계열을 하나의 벡터로 평균화
        global_repr = np.mean(representations, axis=1)
    
    # 쿼리 시계열
    query_repr = global_repr[query_idx]
    
    # 코사인 유사도 계산
    similarities = []
    for i, repr_vec in enumerate(global_repr):
        if i != query_idx:
            sim = 1 - cosine(query_repr, repr_vec)
            similarities.append((i, sim))
    
    # 유사도 순으로 정렬
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print("쿼리 시계열 인덱스: " + str(query_idx))
    print("가장 유사한 " + str(top_k) + "개 시계열:")
    
    for i in range(top_k):
        idx, sim = similarities[i]
        print("  " + str(i+1) + ". 인덱스 " + str(idx) + " (유사도: {:.4f})".format(sim))
    
    return similarities[:top_k]

def run_downstream_tasks(model, test_data):
    """모든 다운스트림 테스크 실행"""
    print("\n" + "="*50)
    print("다운스트림 테스크 실행 시작")
    print("="*50)
    
    # 표현 학습
    print("TS2Vec 표현 추출 중...")
    representations = model.encode(test_data, encoding_window='full_series')
    print("추출된 표현 크기: " + str(representations.shape))
    
    # 1차원 표현을 2차원으로 변환 (sklearn 호환성)
    if representations.ndim == 1:
        representations = representations.reshape(-1, 1)
        print("표현을 2차원으로 변환: " + str(representations.shape))
    
    # 라벨 생성 (패턴 기반)
    labels = create_labeled_dataset(test_data, None)
    
    # 1. 클러스터링
    cluster_labels, silhouette_score = downstream_clustering(representations, n_clusters=3)
    
    # 2. 분류
    accuracy, y_pred, y_test = downstream_classification(representations, labels)
    
    # 3. 유사성 검색
    similar_series = downstream_similarity_search(representations, query_idx=0, top_k=TOP_K_SIMILARITY)
    
    print("\n" + "="*50)
    print("다운스트림 테스크 완료")
    print("="*50)
    print("요약:")
    print("  - 클러스터링 실루엣 점수: {:.4f}".format(silhouette_score))
    print("  - 분류 정확도: {:.4f}".format(accuracy))
    print("  - 유사성 검색: 상위 5개 결과 출력")
    
    return {
        'representations': representations,
        'clustering': {'labels': cluster_labels, 'silhouette_score': silhouette_score},
        'classification': {'accuracy': accuracy, 'predictions': y_pred, 'true_labels': y_test},
        'similarity': similar_series
    }

def test_ts2vec():
    """TS2Vec 모델을 테스트하는 함수"""
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
    
    # TS2Vec 모델 초기화
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("사용하는 디바이스: " + device)
    
    # Loss 최적화를 위한 개선된 하이퍼파라미터 설정
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
    
    # 모델 파일 경로 설정
    model_path = MODEL_PATH
    
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
            # 모델 훈련 - 효율적인 설정
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
    else:
        print("기존 모델이 없습니다. 새로운 모델을 훈련합니다...")
        # 모델 훈련 - 효율적인 설정
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
            return
    
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
