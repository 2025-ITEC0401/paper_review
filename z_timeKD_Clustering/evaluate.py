import pandas as pd
from sklearn.metrics import rand_score, normalized_mutual_info_score

DATASET = ['BasicMotions', 'Epilepsy', 'HandMovementDirection', 'Libras']
# OUTPUT_LEN = [24, 36, 48, 96, 192]
OUTPUT_LEN = [24]
RAWDATA_DIR = './data/rawdata'
RES_DIR = './Result'

def evaluate(kmeans_res_csv, gt_label_csv, label_column_name, seq_len, n_vars):
    try:
        predicted_df = pd.read_csv(kmeans_res_csv)
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
        print(f"File: {kmeans_res_csv}")
        print(f"- ri_score: {ri_score}")
        print(f"- nmi_score: {nmi_score}")
        
    except FileNotFoundError:
        print(f"ERROR: File not found: {gt_label_csv}")
    except KeyError:
        print(f"ERROR: Can't find '{label_column_name}' column.")
    except Exception as e:
        print(f"ERROR: {e}")
        
print("\n\n============= Evaluating =============")
        
for ds in DATASET:
    for output in OUTPUT_LEN:
        kmean_csv = f"{RES_DIR}/csv/{ds}_o{output}_res.csv"
        gt_csv = f"{RAWDATA_DIR}/{ds}_label.csv"
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
        
        evaluate(kmeans_res_csv=kmean_csv, gt_label_csv=gt_csv, label_column_name='label', seq_len=seq_len, n_vars=var)