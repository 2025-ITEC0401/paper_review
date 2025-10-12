# -*- coding: utf-8 -*-
"""
다운스트림 테스크 관련 함수들을 관리하는 모듈
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import (adjusted_rand_score, silhouette_score, 
                           calinski_harabasz_score, davies_bouldin_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
import seaborn as sns
from .config import TOP_K_SIMILARITY
from .data_generator import create_labeled_dataset

def downstream_clustering(representations, n_clusters=3, save_path="result/clustering_results.png"):
    """개선된 클러스터링 다운스트림 테스크"""
    print("\n=== 개선된 시계열 클러스터링 테스트 ===")
    
    # representations가 이미 (n_samples, output_dims) 형태인 경우
    if len(representations.shape) == 2:
        global_repr = representations
    else:
        # 전체 시계열을 하나의 벡터로 평균화 (global pooling)
        global_repr = np.mean(representations, axis=1)  # (n_samples, output_dims)
    
    print("클러스터링 입력 크기: " + str(global_repr.shape))
    
    # 개선된 전처리: PCA 적용 (차원의 80% 유지)
    n_components = int(global_repr.shape[1] * 0.8)
    pca = PCA(n_components=n_components)
    pca_repr = pca.fit_transform(global_repr)
    explained_variance = np.sum(pca.explained_variance_ratio_)
    
    print(f"PCA 차원 축소: {global_repr.shape[1]} -> {n_components}")
    print(f"설명된 분산 비율: {explained_variance:.4f}")
    
    # 개선된 K-means 클러스터링 (더 많은 초기화)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=300)
    cluster_labels = kmeans.fit_predict(pca_repr)
    
    # 다중 평가 메트릭 계산
    silhouette_avg = silhouette_score(pca_repr, cluster_labels)
    calinski_harabasz = calinski_harabasz_score(pca_repr, cluster_labels)
    davies_bouldin = davies_bouldin_score(pca_repr, cluster_labels)
    
    print("개선된 클러스터링 결과:")
    print("  - Number of clusters: " + str(n_clusters))
    print("  - Silhouette score: {:.4f}".format(silhouette_avg))
    print("  - Calinski-Harabasz Index: {:.2f}".format(calinski_harabasz))
    print("  - Davies-Bouldin Index: {:.4f}".format(davies_bouldin))
    
    # 클러스터별 개수
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for i, count in enumerate(counts):
        print("  - Cluster " + str(i) + ": " + str(count) + " samples")
    
    # t-SNE로 시각화 (PCA 적용된 데이터 사용)
    print("Visualizing improved clustering results with t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(pca_repr)-1))
    tsne_result = tsne.fit_transform(pca_repr)
    
    plt.figure(figsize=(12, 8))
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink']
    for i in range(n_clusters):
        mask = cluster_labels == i
        plt.scatter(tsne_result[mask, 0], tsne_result[mask, 1], 
                   c=colors[i % len(colors)], label='Cluster ' + str(i), alpha=0.7, s=50)
    
    plt.title(f'Improved TS2Vec Clustering Results (t-SNE)\nSilhouette Score: {silhouette_avg:.4f}', fontsize=14)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print("Improved clustering visualization saved to " + save_path)
    plt.close()
    
    return cluster_labels, silhouette_avg

def downstream_classification(representations, labels, save_path="result/classification_results.png"):
    """분류 다운스트림 테스크"""
    print("\n=== 시계열 분류 테스트 ===")
    
    # representations가 이미 (n_samples, output_dims) 형태인 경우
    if len(representations.shape) == 2:
        global_repr = representations
    else:
        # 전체 시계열을 하나의 벡터로 평균화
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
