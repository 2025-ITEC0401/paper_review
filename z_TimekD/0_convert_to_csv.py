import os 
import pandas as pd
import numpy as np

ROOT_PATH = "./data"
DATASET = ['BasicMotions', 'Epilepsy', 'HandMovementDirection', 'Libras']

def convert_ts_format(file_path, output_data, output_label, new_dimension, series_length, delimiter):
    all_sample_reshaped = []
    all_labels_expanded = []
    
    print(f"File: {file_path}\n- output(data): {output_data}\n- output(label): {output_label}\n- num_dimenstions: {new_dimension}\n- series_length: {series_length}")
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('@') or line.startswith('#') or line.startswith('%'):
                continue
            
            if (delimiter == ','):
                try:
                    data_part, label = line.rsplit(',', 1)
                    cleaned_data_str = data_part.strip().strip("'")
                    data_str_with_spaces = cleaned_data_str.replace(',', ' ').replace("\\n", ' ')
                    values_str = data_str_with_spaces.split()
                except ValueError:
                    print("ERROR: Invalid Format")
                    continue
            else:
                data_part, label = line.rsplit(':', 1)
                cleaned_data_str = data_part.strip().replace(':', ',')
                values_str = cleaned_data_str.split(',')
            values_flat = [float(v) for v in values_str if v]
            
            if not values_str:
                continue
            
            reshaped_by_dim = np.array(values_flat).reshape(new_dimension, series_length)
            reshaped_by_time = reshaped_by_dim.T
            
            all_sample_reshaped.append(pd.DataFrame(reshaped_by_time))
            all_labels_expanded.extend([label] * series_length)
            
    if not all_sample_reshaped:
        print("!!!! Data does not exist !!!!")
        return
    
    final_df = pd.concat(all_sample_reshaped, ignore_index=True)
    final_df.columns = [f'OT{i + 1}' for i in range(new_dimension)]
    final_df.insert(0, 'date', range(len(final_df)))

    labels_df = pd.DataFrame({
        'date': range(len(all_labels_expanded)),
        'label': all_labels_expanded
    })
        
    final_df.to_csv(output_data, index=False)
    labels_df.to_csv(output_label, index=False)
    
    print(f"\nComplete!\n- Data: {output_data},\n- Label: {output_label}\n")

def merge_CSV(trainCSV, testCSV, targetPath):
    print("Merging csv file for Train and test...")
    df_train = pd.read_csv(trainCSV)
    df_test = pd.read_csv(testCSV)

    combined_df = pd.concat([df_train, df_test], ignore_index=True)
    num_rows = len(combined_df)
    date_range = pd.date_range(start='2023-01-01', periods=num_rows, freq='H')

    first_col_name = combined_df.columns[0]
    combined_df[first_col_name] = date_range

    combined_df.to_csv(targetPath, index=False)
    print(f"Complete!\n- Path: {targetPath}\n---------------------------\n")
            
for ds in DATASET:
    path = f"{ROOT_PATH}/{ds}"
    match ds:
        case 'BasicMotions':
            new_dimension = 6
            series_length = 100
        case 'Epilepsy':
            new_dimension = 3
            series_length = 206
        case 'HandMovementDirection':
            new_dimension = 10
            series_length = 400
        case 'Libras':
            new_dimension = 2
            series_length = 45
        case _:
            print("!!! Invalid Dataset !!!")
            continue    
        
    for tp in ['TRAIN', 'TEST']:
        print(f"\n============ {ds}_{tp} ============")
        for extension in ['arff', 'ts']:
            if (extension == 'arff'):
                delimiter = ','
            elif (extension == 'ts'):
                delimiter = ':'
            else:
                print("!!! Invalid Data Format !!!")
                continue
            
            filename = f"{path}/{ds}_{tp}.{extension}"

            if (not os.path.exists(filename)):
                print(f"File: {filename}")
                print(" !! File not Found !!\n")
                continue
            
            if (extension == 'ts' and os.path.exists(f"{path}/{ds}_{tp}.arff")):
                print(f"File: {filename}")
                print("!! .arff file exists... Skip...\n")
                continue               
            
            convert_ts_format(
                file_path=filename,
                output_data=f"{ROOT_PATH}/{ds}_{tp}_data.csv",
                output_label=f"{ROOT_PATH}/{ds}_{tp}_label.csv",
                new_dimension=new_dimension,
                series_length=series_length,
                delimiter=delimiter
            )
    merge_CSV(trainCSV=f"{ROOT_PATH}/{ds}_TRAIN_data.csv", testCSV=f"{ROOT_PATH}/{ds}_TEST_data.csv", targetPath = f"{ROOT_PATH}/data/{ds}.csv")
    merge_CSV(trainCSV=f"{ROOT_PATH}/{ds}_TRAIN_label.csv", testCSV=f"{ROOT_PATH}/{ds}_TEST_label.csv", targetPath = f"{ROOT_PATH}/{ds}_label.csv")
