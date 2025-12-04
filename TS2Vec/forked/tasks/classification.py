import numpy as np
from . import _eval_protocols as eval_protocols
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

def eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='linear'):
    assert train_labels.ndim == 1 or train_labels.ndim == 2
    train_repr = model.encode(train_data, encoding_window='full_series' if train_labels.ndim == 1 else None)
    test_repr = model.encode(test_data, encoding_window='full_series' if train_labels.ndim == 1 else None)

    if eval_protocol == 'linear':
        fit_clf = eval_protocols.fit_lr
    elif eval_protocol == 'svm':
        fit_clf = eval_protocols.fit_svm
    elif eval_protocol == 'knn':
        fit_clf = eval_protocols.fit_knn
    else:
        assert False, 'unknown evaluation protocol'

    def merge_dim01(array):
        return array.reshape(array.shape[0]*array.shape[1], *array.shape[2:])

    if train_labels.ndim == 2:
        train_repr = merge_dim01(train_repr)
        train_labels = merge_dim01(train_labels)
        test_repr = merge_dim01(test_repr)
        test_labels = merge_dim01(test_labels)

    clf = fit_clf(train_repr, train_labels)

    acc = clf.score(test_repr, test_labels)
    y_pred = clf.predict(test_repr)
    
    # F1 scores
    f1_macro = f1_score(test_labels, y_pred, average='macro')
    f1_weighted = f1_score(test_labels, y_pred, average='weighted')
    
    # Cross-validation accuracy
    # 클래스당 최소 샘플 수를 확인하여 적절한 fold 수 결정
    unique_labels, label_counts = np.unique(train_labels, return_counts=True)
    min_samples_per_class = label_counts.min()
    n_splits = min(5, min_samples_per_class)  # 최대 5-fold, 최소 클래스 샘플 수 이하
    
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, train_repr, train_labels, cv=cv, scoring='accuracy')
        cv_acc = cv_scores.mean()
    else:
        # 샘플이 너무 적으면 CV 수행 불가
        cv_acc = acc  # 테스트 정확도로 대체
    
    # Number of classes
    n_classes = len(np.unique(train_labels))
    
    if eval_protocol == 'linear':
        y_score = clf.predict_proba(test_repr)
    else:
        y_score = clf.decision_function(test_repr)
    test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max()+1))
    auprc = average_precision_score(test_labels_onehot, y_score)
    
    return y_score, { 
        'acc': acc, 
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'cv_acc': cv_acc,
        'n_classes': n_classes,
        'auprc': auprc 
    }
