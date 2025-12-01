import h5py
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

ROOT_DIR = '../data'
# DATASET = ['AtrialFibrillation', 'PEMS-SF', 'StandWalkJump']
DATASET = ['AtrialFibrillation', 'StandWalkJump']
# OUTPUT_LEN_LIST = [24, 36, 48, 96, 192]
OUTPUT_LEN_LIST = [24]
TYPE = ['train', 'val']
RES_DIR = '../Result_csv'
KEY = 'embeddings'
CONTAMINATION = 0.05

os.makedirs(RES_DIR, exist_ok=True)


def run_isolation_forest(train_file, test_file, output_csv):
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
    
    result_df = pd.DataFrame({
        'window_index': range(len(val_data)),
        'anomaly_score': val_scores,
        'is_anomaly': (val_scores > threshold).astype(int),
        'threshold_used': threshold
    })
    
    result_df.to_csv(output_csv, index=False)
    print(f"Result Saved: {output_csv}\n\n")



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
        
        run_isolation_forest(h5_train_path, h5_test_path, f"{RES_DIR}/{ds}_o{output_len}_anomaly_detection_threshold_{CONTAMINATION}_res.csv")

