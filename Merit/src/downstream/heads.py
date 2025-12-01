# src/downstream/heads.py
# 다운스트림 태스크 헤드 모듈

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, OneClassSVM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    normalized_mutual_info_score, adjusted_rand_score, rand_score, silhouette_score,
    roc_auc_score, mean_squared_error, mean_absolute_error
)
import warnings
warnings.filterwarnings('ignore')


class ClassificationHead:
    """Classification 다운스트림 태스크"""

    def __init__(self, method='svm'):
        self.method = method
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

    def _get_model(self):
        if self.method == 'svm':
            return SVC(kernel='rbf', C=1.0, random_state=42)
        elif self.method == 'knn':
            return KNeighborsClassifier(n_neighbors=5)
        elif self.method == 'logistic':
            return LogisticRegression(max_iter=1000, random_state=42)
        elif self.method == 'rf':
            return RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def evaluate(self, representations, labels, test_size=0.2):
        """Classification 평가"""
        labels_encoded = self.label_encoder.fit_transform(labels)

        X_train, X_test, y_train, y_test = train_test_split(
            representations, labels_encoded,
            test_size=test_size, random_state=42, stratify=labels_encoded
        )

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.model = self._get_model()
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'num_classes': len(self.label_encoder.classes_),
            'train_size': len(X_train),
            'test_size': len(X_test)
        }

        return results

    def cross_validate(self, representations, labels, cv=5):
        """Cross-validation 평가"""
        labels_encoded = self.label_encoder.fit_transform(labels)
        X_scaled = self.scaler.fit_transform(representations)

        model = self._get_model()
        scores = cross_val_score(model, X_scaled, labels_encoded, cv=cv, scoring='accuracy')

        return {
            'cv_accuracy_mean': scores.mean(),
            'cv_accuracy_std': scores.std(),
            'cv_scores': scores.tolist()
        }


class ClusteringHead:
    """Clustering 다운스트림 태스크"""

    def __init__(self, n_clusters=None):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()

    def evaluate(self, representations, labels):
        """Clustering 평가"""
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        n_clusters = self.n_clusters or len(np.unique(labels_encoded))

        X_scaled = self.scaler.fit_transform(representations)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)

        results = {
            'nmi': normalized_mutual_info_score(labels_encoded, cluster_labels),
            'ri': rand_score(labels_encoded, cluster_labels),
            'ari': adjusted_rand_score(labels_encoded, cluster_labels),
            'silhouette': silhouette_score(X_scaled, cluster_labels),
            'n_clusters': n_clusters,
            'inertia': kmeans.inertia_
        }

        return results, cluster_labels, kmeans.cluster_centers_


class AnomalyDetectionHead:
    """Anomaly Detection 다운스트림 태스크"""

    def __init__(self, method='isolation_forest', contamination=0.1):
        self.method = method
        self.contamination = contamination
        self.scaler = StandardScaler()

    def _get_model(self):
        if self.method == 'isolation_forest':
            return IsolationForest(contamination=self.contamination, random_state=42)
        elif self.method == 'one_class_svm':
            return OneClassSVM(nu=self.contamination, kernel='rbf')
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def evaluate(self, representations, labels=None):
        """Anomaly Detection 평가"""
        X_scaled = self.scaler.fit_transform(representations)
        model = self._get_model()

        if labels is not None:
            label_encoder = LabelEncoder()
            labels_encoded = label_encoder.fit_transform(labels)
            unique, counts = np.unique(labels_encoded, return_counts=True)
            anomaly_class = unique[np.argmin(counts)]

            normal_mask = labels_encoded != anomaly_class
            X_normal = X_scaled[normal_mask]

            model.fit(X_normal)

            predictions = model.predict(X_scaled)
            predictions = (predictions == -1).astype(int)
            true_labels = (labels_encoded == anomaly_class).astype(int)

            results = {
                'accuracy': accuracy_score(true_labels, predictions),
                'precision': precision_score(true_labels, predictions, zero_division=0),
                'recall': recall_score(true_labels, predictions, zero_division=0),
                'f1': f1_score(true_labels, predictions, zero_division=0),
                'anomaly_ratio': float(np.mean(true_labels)),
                'detected_ratio': float(np.mean(predictions))
            }

            if hasattr(model, 'decision_function'):
                scores = -model.decision_function(X_scaled)
                results['roc_auc'] = roc_auc_score(true_labels, scores)
        else:
            model.fit(X_scaled)
            predictions = model.predict(X_scaled)
            predictions = (predictions == -1).astype(int)

            results = {
                'detected_anomaly_ratio': float(np.mean(predictions)),
                'num_anomalies': int(np.sum(predictions)),
                'num_normal': int(np.sum(predictions == 0))
            }

        return results


class ForecastingHead:
    """Forecasting 다운스트림 태스크 (Representation 기반)"""

    def __init__(self, horizon=1):
        self.horizon = horizon
        self.scaler = StandardScaler()

    def evaluate(self, representations, original_data=None):
        """Representation 기반 예측 평가"""
        X_scaled = self.scaler.fit_transform(representations)

        X_train = X_scaled[:-self.horizon]
        y_train = X_scaled[self.horizon:]

        split_idx = int(len(X_train) * 0.8)
        X_tr, X_te = X_train[:split_idx], X_train[split_idx:]
        y_tr, y_te = y_train[:split_idx], y_train[split_idx:]

        model = Ridge(alpha=1.0)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        mse = mean_squared_error(y_te, y_pred)
        mae = mean_absolute_error(y_te, y_pred)

        ss_res = np.sum((y_te - y_pred) ** 2)
        ss_tot = np.sum((y_te - np.mean(y_te, axis=0)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        results = {
            'mse': float(mse),
            'rmse': float(np.sqrt(mse)),
            'mae': float(mae),
            'r2': float(r2),
            'horizon': self.horizon,
            'train_size': len(X_tr),
            'test_size': len(X_te)
        }

        return results


class ImputationHead:
    """Imputation 다운스트림 태스크 (Representation 기반 결측치 복원)"""

    def __init__(self, mask_ratio=0.2):
        self.mask_ratio = mask_ratio
        self.scaler = StandardScaler()

    def evaluate(self, representations):
        """Imputation 평가 - representation의 일부를 마스킹하고 복원 능력 측정"""
        X_scaled = self.scaler.fit_transform(representations)

        # 데이터 분할
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]

        # 테스트 데이터에 마스크 적용
        np.random.seed(42)
        mask = np.random.random(X_test.shape) < self.mask_ratio
        X_test_masked = X_test.copy()
        X_test_masked[mask] = 0  # 마스킹된 부분을 0으로 설정

        # KNN 기반 imputation (학습 데이터의 평균으로 복원)
        from sklearn.neighbors import KNeighborsRegressor

        # 각 feature에 대해 KNN으로 복원
        X_imputed = X_test_masked.copy()
        knn = KNeighborsRegressor(n_neighbors=5)

        for feat_idx in range(X_test.shape[1]):
            # 마스킹되지 않은 feature들을 사용해 마스킹된 feature 예측
            feat_mask = mask[:, feat_idx]
            if np.sum(feat_mask) == 0:
                continue

            # 마스킹되지 않은 샘플로 학습
            train_mask = ~mask[:, feat_idx]
            if np.sum(train_mask) < 5:
                continue

            other_feats = np.delete(np.arange(X_test.shape[1]), feat_idx)
            knn.fit(X_test_masked[train_mask][:, other_feats], X_test[train_mask, feat_idx])

            # 마스킹된 샘플 예측
            X_imputed[feat_mask, feat_idx] = knn.predict(X_test_masked[feat_mask][:, other_feats])

        # 마스킹된 부분에 대해서만 메트릭 계산
        true_values = X_test[mask]
        pred_values = X_imputed[mask]

        mse = mean_squared_error(true_values, pred_values)
        mae = mean_absolute_error(true_values, pred_values)

        results = {
            'mse': float(mse),
            'rmse': float(np.sqrt(mse)),
            'mae': float(mae),
            'mask_ratio': self.mask_ratio,
            'num_masked': int(np.sum(mask)),
            'test_size': len(X_test)
        }

        return results
