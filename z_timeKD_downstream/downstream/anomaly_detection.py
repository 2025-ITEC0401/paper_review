import h5py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

ROOT_DIR = '../data'
# DATASET = ['AtrialFibrillation', 'PEMS-SF', 'StandWalkJump']
DATASET = ['AtrialFibrillation', 'StandWalkJump']
# OUTPUT_LEN_LIST = [24, 36, 48, 96, 192]
OUTPUT_LEN_LIST = [24]
TYPE = ['train', 'val']
RES_DIR = '../Result_csv'
FIG_DIR = '../Result_fig'
KEY = 'embeddings'
CONTAMINATION = 0.05

os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

def run_isolation_forest(train_file, test_file, output_csv, output_figure):
    print("Loading Data from Embeddings...")
    print(f"- Train: {train_file}")
    print(f"- Val: {test_file}")
    
    with h5py.File(train_file, 'r') as f:
        train_data = f[KEY][:]
    
    with h5py.File(test_file, 'r') as g:
        val_data = g[KEY][:]
    
    print("Completed.\n\nTraining Isolation Forest...")
    iso_forest = IsolationForest(n_estimators=100,
                                 contamination='auto',
                                 random_state=42,
                                 n_jobs=1)
    iso_forest.fit(train_data)
    
    train_scores = -1 * iso_forest.decision_function(train_data)
    val_scores = -1 * iso_forest.decision_function(val_data)
    
    threshold = np.percentile(train_scores, 100 * (1 - CONTAMINATION))
    
    print(f"\nSetting Threshold... (Top {CONTAMINATION * 100}%)")
    print(f"- Threshold Score: {threshold:.4f}")
    
    anomalies_idx = np.where(val_scores > threshold)[0]
    
    print("\n[Result]")
    print(f"- The number of total validation data: {len(val_data)}")
    print(f"- The number of found anormal data: {len(anomalies_idx)}")
    print(f"- Anormal data ratio: {len(anomalies_idx) / len(val_data) * 100:.2f}%\n\n")
    
    result_df = pd.DataFrame({
        'window_index': range(len(val_data)),
        'anomaly_score': val_scores,
        'is_anomaly': (val_scores > threshold).astype(int),
        'threshold_used': threshold
    })
    
    result_df.to_csv(output_csv, index=False)
    print(f"Result Saved: {output_csv}\n")
    
    plt.figure(figsize=(14, 10))
    
    # (A) 점수 분포 비교 (Histogram)
    plt.subplot(2, 1, 1)
    sns.histplot(train_scores, label='Train Scores (Baseline)', color='blue', alpha=0.3, kde=True)
    sns.histplot(val_scores, label='Val Scores (Target)', color='orange', alpha=0.3, kde=True)
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.2f})')
    plt.title('Anomaly Score Distribution (Train vs Val)')
    plt.xlabel('Anomaly Score (Higher = More Anomalous)')
    plt.legend()

    # (B) 시계열 흐름상 이상치 위치 (Val Data)
    plt.subplot(2, 1, 2)
    plt.plot(val_scores, label='Val Anomaly Score', color='black', alpha=0.7, linewidth=1)
    plt.scatter(anomalies_idx, val_scores[anomalies_idx], color='red', s=30, label='Detected Anomaly', zorder=5)
    plt.axhline(threshold, color='red', linestyle='--', alpha=0.5)
    plt.title('Anomaly Detection Result on Validation Sequence')
    plt.xlabel('Time Window Index')
    plt.ylabel('Anomaly Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_figure)


print("\n\n========== Anomaly Detection ==========")

idx = 1
for ds in DATASET:
    for output_len in OUTPUT_LEN_LIST:
        h5_train_path = f"{ROOT_DIR}/{ds}_o{output_len}_{TYPE[0]}_consolidated.h5"
        h5_test_path = f"{ROOT_DIR}/{ds}_o{output_len}_{TYPE[1]}_consolidated.h5"
        
        if not os.path.exists(h5_train_path):
            print(f"({idx}/{len(DATASET) * len(OUTPUT_LEN_LIST)}) File not found: {h5_train_path}")
            idx += 1
            continue
        
        if not os.path.exists(h5_test_path):
            print(f"({idx}/{len(DATASET) * len(OUTPUT_LEN_LIST)}) File not found: {h5_test_path}")
            idx += 1
            continue
        
        print(f"({idx}/{len(DATASET) * len(OUTPUT_LEN_LIST)}) Target: {ds}_o{output_len}\n")
        idx += 1
        
        run_isolation_forest(h5_train_path, h5_test_path, f"{RES_DIR}/{ds}_o{output_len}_anomaly_detection_threshold_{CONTAMINATION}_res.csv", f"{FIG_DIR}/{ds}_o{output_len}_anomaly_detection_threshold_{CONTAMINATION}_res.png")

