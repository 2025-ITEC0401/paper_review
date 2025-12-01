import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = '../data'
# DATASET = ['AtrialFibrillation', 'PEMS-SF', 'StandWalkJump']
DATASET = ['AtrialFibrillation', 'StandWalkJump']
# OUTPUT_LEN_LIST = [24, 36, 48, 96, 192]
OUTPUT_LEN_LIST = [24]
RES_DIR = '../Result_csv'
FIG_DIR = '../Result_fig'
CONTAMINATION = 0.05

os.makedirs(FIG_DIR, exist_ok=True)

def evaluate(raw_csv_path, res_txt_path, res_fig_path):
    if not os.path.exists(raw_csv_path):
        print(f"File not Found: {raw_csv_path}\n")
        return
    df = pd.read_csv(raw_csv_path)
    
    val_score = df['anomaly_score'].values
    anomalies_idx = df.index[df['is_anomaly'] == 1].tolist()
    total_count = len(df)
    anomaly_count = len(df[df['is_anomaly'] == 1])
    ratio = (anomaly_count / total_count * 100) if total_count > 0 else 0
    if 'threshold_used' in df.columns:
        threshold_val = df['threshold_used'].iloc[0]
    else:
        threshold_val = 0.0
        
    print(f"File: {raw_csv_path}")
    print(f"- The number of total validation data: {total_count}")
    print(f"- The number of found anormal data: {anomaly_count}")
    print(f"- Threshold Score: {threshold_val:.4f}")
    print(f"- Anormal data ratio: {ratio:.2f}%\n")
    
    with open(res_txt_path, "w", encoding='utf-8') as f:
        f.write(f"File: {raw_csv_path}\n")
        f.write(f"- The number of total validation data: {total_count}\n")
        f.write(f"- The number of found anormal data: {anomaly_count}\n")
        f.write(f"- Threshold Score: {threshold_val:.4f}\n")
        f.write(f"- Anormal data ratio: {ratio:.2f}%\n")
    
    plt.figure(figsize=(14, 10))
    
    # (A) 점수 분포 비교 (Histogram)
    plt.subplot(2, 1, 1)
    sns.histplot(val_score, label='Val Scores (Target)', color='orange', alpha=0.3, kde=True)
    plt.axvline(threshold_val, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold_val:.2f})')
    plt.title('Anomaly Score Distribution (Train vs Val)')
    plt.xlabel('Anomaly Score (Higher = More Anomalous)')
    plt.legend()

    # (B) 시계열 흐름상 이상치 위치 (Val Data)
    plt.subplot(2, 1, 2)
    plt.plot(val_score, label='Val Anomaly Score', color='black', alpha=0.7, linewidth=1)
    plt.scatter(anomalies_idx, val_score[anomalies_idx], color='red', s=30, label='Detected Anomaly', zorder=5)
    plt.axhline(threshold_val, color='red', linestyle='--', alpha=0.5)
    plt.title('Anomaly Detection Result on Validation Sequence')
    plt.xlabel('Time Window Index')
    plt.ylabel('Anomaly Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig(res_fig_path)
    
    
    
print("\n\n========== Anomaly Detection ==========")

idx = 1
for ds in DATASET:
    for o in OUTPUT_LEN_LIST:
        raw_csv_path = f"{RES_DIR}/{ds}_o{o}_anomaly_detection_threshold_{CONTAMINATION}_res.csv"
        res_txt_path = f"{RES_DIR}/{ds}_o{o}_anomaly_detection_threshold_{CONTAMINATION}_summary.txt"
        res_fig_path = f"{FIG_DIR}/{ds}_o{o}_anomaly_detection_threshold_{CONTAMINATION}_res.png"
        
        if not os.path.exists(raw_csv_path):
            print(f"({idx}/{len(DATASET) * len(OUTPUT_LEN_LIST)}) File not found: {raw_csv_path}")
            idx += 1
            continue
        
        print(f"({idx}/{len(DATASET) * len(OUTPUT_LEN_LIST)}) Target: {ds}_o{o}")
        idx += 1
        
        evaluate(raw_csv_path=raw_csv_path, res_txt_path=res_txt_path, res_fig_path=res_fig_path)

