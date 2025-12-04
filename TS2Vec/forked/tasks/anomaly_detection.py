import numpy as np
import time
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import bottleneck as bn

# consider delay threshold and missing segments
def get_range_proba(predict, label, delay=7):
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    new_predict = np.array(predict)
    pos = 0

    for sp in splits:
        if is_anomaly:
            if 1 in predict[pos:min(pos + delay + 1, sp)]:
                new_predict[pos: sp] = 1
            else:
                new_predict[pos: sp] = 0
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)

    if is_anomaly:  # anomaly in the end
        if 1 in predict[pos: min(pos + delay + 1, sp)]:
            new_predict[pos: sp] = 1
        else:
            new_predict[pos: sp] = 0

    return new_predict


# set missing = 0
def reconstruct_label(timestamp, label):
    timestamp = np.asarray(timestamp, np.int64)
    index = np.argsort(timestamp)

    timestamp_sorted = np.asarray(timestamp[index])
    interval = np.min(np.diff(timestamp_sorted))

    label = np.asarray(label, np.int64)
    label = np.asarray(label[index])

    idx = (timestamp_sorted - timestamp_sorted[0]) // interval

    new_label = np.zeros(shape=((timestamp_sorted[-1] - timestamp_sorted[0]) // interval + 1,), dtype=np.int)
    new_label[idx] = label

    return new_label


def eval_ad_result(test_pred_list, test_labels_list, test_timestamps_list, delay, test_scores_list=None):
    labels = []
    pred = []
    scores = []
    for i, (test_pred, test_labels, test_timestamps) in enumerate(zip(test_pred_list, test_labels_list, test_timestamps_list)):
        assert test_pred.shape == test_labels.shape == test_timestamps.shape
        test_labels = reconstruct_label(test_timestamps, test_labels)
        test_pred = reconstruct_label(test_timestamps, test_pred)
        test_pred = get_range_proba(test_pred, test_labels, delay)
        labels.append(test_labels)
        pred.append(test_pred)
        if test_scores_list is not None:
            test_scores = test_scores_list[i]
            # Reconstruct scores similarly
            timestamp = np.asarray(test_timestamps, np.int64)
            index = np.argsort(timestamp)
            timestamp_sorted = np.asarray(timestamp[index])
            interval = np.min(np.diff(timestamp_sorted))
            test_scores_sorted = np.asarray(test_scores[index])
            idx = (timestamp_sorted - timestamp_sorted[0]) // interval
            new_scores = np.zeros(shape=((timestamp_sorted[-1] - timestamp_sorted[0]) // interval + 1,), dtype=np.float64)
            new_scores[idx] = test_scores_sorted
            scores.append(new_scores)
    
    labels = np.concatenate(labels)
    pred = np.concatenate(pred)
    
    result = {
        'acc': round(float(accuracy_score(labels, pred)), 4),
        'precision': round(float(precision_score(labels, pred, zero_division=0)), 4),
        'recall': round(float(recall_score(labels, pred, zero_division=0)), 4),
        'f1': round(float(f1_score(labels, pred, zero_division=0)), 4),
    }
    
    # ROC-AUC 계산 (scores가 있는 경우)
    if test_scores_list is not None and len(scores) > 0:
        scores = np.concatenate(scores)
        # labels에 양성/음성 클래스가 모두 있어야 ROC-AUC 계산 가능
        if len(np.unique(labels)) > 1:
            result['roc_auc'] = round(float(roc_auc_score(labels, scores)), 4)
        else:
            result['roc_auc'] = 0.0
    else:
        # scores가 없으면 pred를 사용
        if len(np.unique(labels)) > 1:
            result['roc_auc'] = round(float(roc_auc_score(labels, pred)), 4)
        else:
            result['roc_auc'] = 0.0
    
    return result


def np_shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def eval_anomaly_detection(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay):
    t = time.time()
    
    all_train_repr = {}
    all_test_repr = {}
    all_train_repr_wom = {}
    all_test_repr_wom = {}
    for k in all_train_data:
        train_data = all_train_data[k]
        test_data = all_test_data[k]
        
        # 데이터 형태 확인: 1D면 (timesteps,), 2D면 (timesteps, features)
        concat_data = np.concatenate([train_data, test_data], axis=0)
        if concat_data.ndim == 1:
            # 단변량 데이터: (timesteps,) -> (1, timesteps, 1)
            concat_data = concat_data.reshape(1, -1, 1)
        else:
            # 다변량 데이터: (timesteps, features) -> (1, timesteps, features)
            concat_data = np.expand_dims(concat_data, axis=0)

        full_repr = model.encode(
            concat_data,
            mask='mask_last',
            causal=True,
            sliding_length=1,
            sliding_padding=200,
            batch_size=256
        ).squeeze()
        all_train_repr[k] = full_repr[:len(train_data)]
        all_test_repr[k] = full_repr[len(train_data):]

        full_repr_wom = model.encode(
            concat_data,
            causal=True,
            sliding_length=1,
            sliding_padding=200,
            batch_size=256
        ).squeeze()
        all_train_repr_wom[k] = full_repr_wom[:len(train_data)]
        all_test_repr_wom[k] = full_repr_wom[len(train_data):]
    
    # =====================================================================
    # Method 1: TS2Vec (Original - Error-based)
    # =====================================================================
    res_log_ts2vec = []
    labels_log = []
    timestamps_log = []
    scores_log_ts2vec = []
    for k in all_train_data:
        train_data = all_train_data[k]
        train_labels = all_train_labels[k]
        train_timestamps = all_train_timestamps[k]

        test_data = all_test_data[k]
        test_labels = all_test_labels[k]
        test_timestamps = all_test_timestamps[k]

        train_err = np.abs(all_train_repr_wom[k] - all_train_repr[k]).sum(axis=1)
        test_err = np.abs(all_test_repr_wom[k] - all_test_repr[k]).sum(axis=1)

        ma = np_shift(bn.move_mean(np.concatenate([train_err, test_err]), 21), 1)
        train_err_adj = (train_err - ma[:len(train_err)]) / ma[:len(train_err)]
        test_err_adj = (test_err - ma[len(train_err):]) / ma[len(train_err):]
        train_err_adj = train_err_adj[22:]

        thr = np.mean(train_err_adj) + 4 * np.std(train_err_adj)
        test_res = (test_err_adj > thr) * 1

        for i in range(len(test_res)):
            if i >= delay and test_res[i-delay:i].sum() >= 1:
                test_res[i] = 0

        res_log_ts2vec.append(test_res)
        labels_log.append(test_labels)
        timestamps_log.append(test_timestamps)
        scores_log_ts2vec.append(test_err_adj)
    
    # =====================================================================
    # Method 2: Isolation Forest
    # =====================================================================
    res_log_iforest = []
    scores_log_iforest = []
    for k in all_train_data:
        train_repr = all_train_repr[k]
        test_repr = all_test_repr[k]
        test_labels = all_test_labels[k]
        
        # Isolation Forest 학습 (정상 데이터로만)
        iforest = IsolationForest(n_estimators=100, contamination='auto', random_state=42, n_jobs=-1)
        iforest.fit(train_repr)
        
        # 예측: -1은 이상, 1은 정상 -> 0은 정상, 1은 이상으로 변환
        test_pred = iforest.predict(test_repr)
        test_pred = (test_pred == -1).astype(int)
        
        # anomaly score (낮을수록 이상)
        test_scores = -iforest.score_samples(test_repr)  # 부호 반전하여 높을수록 이상
        
        # delay 처리
        for i in range(len(test_pred)):
            if i >= delay and test_pred[i-delay:i].sum() >= 1:
                test_pred[i] = 0
        
        res_log_iforest.append(test_pred)
        scores_log_iforest.append(test_scores)
    
    # =====================================================================
    # Method 3: One-Class SVM
    # =====================================================================
    res_log_ocsvm = []
    scores_log_ocsvm = []
    for k in all_train_data:
        train_repr = all_train_repr[k]
        test_repr = all_test_repr[k]
        test_labels = all_test_labels[k]
        
        # One-Class SVM 학습 (정상 데이터로만)
        # 데이터가 많으면 샘플링
        if len(train_repr) > 5000:
            indices = np.random.choice(len(train_repr), 5000, replace=False)
            train_repr_sampled = train_repr[indices]
        else:
            train_repr_sampled = train_repr
        
        ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
        ocsvm.fit(train_repr_sampled)
        
        # 예측: -1은 이상, 1은 정상 -> 0은 정상, 1은 이상으로 변환
        test_pred = ocsvm.predict(test_repr)
        test_pred = (test_pred == -1).astype(int)
        
        # anomaly score (decision function의 부호 반전)
        test_scores = -ocsvm.decision_function(test_repr)  # 높을수록 이상
        
        # delay 처리
        for i in range(len(test_pred)):
            if i >= delay and test_pred[i-delay:i].sum() >= 1:
                test_pred[i] = 0
        
        res_log_ocsvm.append(test_pred)
        scores_log_ocsvm.append(test_scores)
    
    t = time.time() - t
    
    # 각 방법별 결과 계산
    eval_res_ts2vec = eval_ad_result(res_log_ts2vec, labels_log, timestamps_log, delay, scores_log_ts2vec)
    eval_res_iforest = eval_ad_result(res_log_iforest, labels_log, timestamps_log, delay, scores_log_iforest)
    eval_res_ocsvm = eval_ad_result(res_log_ocsvm, labels_log, timestamps_log, delay, scores_log_ocsvm)
    
    eval_res = {
        'ts2vec': eval_res_ts2vec,
        'isolation_forest': eval_res_iforest,
        'one_class_svm': eval_res_ocsvm,
        'infer_time': round(t, 4)
    }
    
    res_log = {
        'ts2vec': res_log_ts2vec,
        'isolation_forest': res_log_iforest,
        'one_class_svm': res_log_ocsvm
    }
    
    return res_log, eval_res


def eval_anomaly_detection_coldstart(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay):
    t = time.time()
    
    all_data = {}
    all_repr = {}
    all_repr_wom = {}
    for k in all_train_data:
        all_data[k] = np.concatenate([all_train_data[k], all_test_data[k]])
        all_repr[k] = model.encode(
            all_data[k].reshape(1, -1, 1),
            mask='mask_last',
            causal=True,
            sliding_length=1,
            sliding_padding=200,
            batch_size=256
        ).squeeze()
        all_repr_wom[k] = model.encode(
            all_data[k].reshape(1, -1, 1),
            causal=True,
            sliding_length=1,
            sliding_padding=200,
            batch_size=256
        ).squeeze()
        
    res_log = []
    labels_log = []
    timestamps_log = []
    scores_log = []
    for k in all_data:
        data = all_data[k]
        labels = np.concatenate([all_train_labels[k], all_test_labels[k]])
        timestamps = np.concatenate([all_train_timestamps[k], all_test_timestamps[k]])
        
        err = np.abs(all_repr_wom[k] - all_repr[k]).sum(axis=1)
        ma = np_shift(bn.move_mean(err, 21), 1)
        err_adj = (err - ma) / ma
        
        MIN_WINDOW = len(data) // 10
        thr = bn.move_mean(err_adj, len(err_adj), MIN_WINDOW) + 4 * bn.move_std(err_adj, len(err_adj), MIN_WINDOW)
        res = (err_adj > thr) * 1
        
        for i in range(len(res)):
            if i >= delay and res[i-delay:i].sum() >= 1:
                res[i] = 0

        res_log.append(res[MIN_WINDOW:])
        labels_log.append(labels[MIN_WINDOW:])
        timestamps_log.append(timestamps[MIN_WINDOW:])
        scores_log.append(err_adj[MIN_WINDOW:])  # anomaly score 저장
    t = time.time() - t
    
    eval_res = eval_ad_result(res_log, labels_log, timestamps_log, delay, scores_log)
    eval_res['infer_time'] = t
    return res_log, eval_res

