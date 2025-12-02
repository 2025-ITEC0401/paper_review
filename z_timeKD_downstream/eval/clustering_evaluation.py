import pandas as pd
from sklearn.metrics import rand_score, normalized_mutual_info_score

DATASET = ['ArticularyWordRecognition', 'AtrialFibrillation', 'NATOPS', 'PenDigits', 'StandWalkJump', 'UWaveGestureLibrary']
# OUTPUT_LEN = [24, 36, 48, 96, 192]
OUTPUT_LEN = [24]
RAWDATA_DIR = '../data'
RES_DIR = '../Result_csv'

def evaluate(pred_res_csv, gt_label_csv, label_column_name, seq_len, res_txt_path):
    try:
        predicted_df = pd.read_csv(pred_res_csv)
        predicted_labels_final = predicted_df['cluster'].values
        
        ground_truth_df = pd.read_csv(gt_label_csv)
        num_train = int(len(ground_truth_df) * 0.7)
        num_test = int(len(ground_truth_df) * 0.2)
        num_vali = len(ground_truth_df) - num_train - num_test
        
        border1_val = num_train - seq_len
        border2_val = num_train + num_vali
        
        true_labels_slice = ground_truth_df[label_column_name].iloc[border1_val:border2_val]
        true_labels_for_eval = true_labels_slice.iloc[:len(predicted_labels_final)].values

        if (len(predicted_labels_final) != len(true_labels_for_eval)):
            print(f"ERROR: len(pred_lables) - {int(len(predicted_labels_final) / 6)} != len(true_labels_for_eval) - {len(true_labels_for_eval)}")
            return

        ri_score = rand_score(true_labels_for_eval, predicted_labels_final)
        nmi_score = normalized_mutual_info_score(true_labels_for_eval, predicted_labels_final)
        print(f"File: {pred_res_csv}")
        print(f"- ri_score: {ri_score}")
        print(f"- nmi_score: {nmi_score}\n")

        with open(res_txt_path, "w", encoding='utf-8') as f:
            f.write(f"File: {pred_csv}\n")
            f.write(f"- ri_score: {ri_score}\n")
            f.write(f"- nmi_score: {nmi_score}\n\n")
            
        
    except FileNotFoundError:
        print(f"ERROR: File not found: {gt_label_csv}")
    except KeyError:
        print(f"ERROR: Can't find '{label_column_name}' column.")
    except Exception as e:
        print(f"ERROR: {e}")
        
print("\n\n============= Evaluating =============")
        
for ds in DATASET:
    for output in OUTPUT_LEN:
        for method in ['kmeans', 'spectral']:
            pred_csv = f"{RES_DIR}/{ds}_o{output}_clustering_{method}_res.csv"
            gt_csv = f"{RAWDATA_DIR}/{ds}_label.csv"
            summary_txt = f"{RES_DIR}/{ds}_o{output}_clustering_{method}_summary.txt"
            seq_len = 96
            
            match ds:
                case 'BasicMotions':
                    var = 6
                case 'Epilepsy':
                    var = 3
                case 'HandMovementDirection':
                    var = 10
                case 'Libras':
                    var = 2
            
            evaluate(pred_res_csv=pred_csv, gt_label_csv=gt_csv, label_column_name='label', seq_len=seq_len, res_txt_path=summary_txt)