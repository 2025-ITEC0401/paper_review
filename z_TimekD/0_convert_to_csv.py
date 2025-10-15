import os
from glob import glob
from sktime.datasets import load_from_arff_to_dataframe, load_from_tsfile
import pandas as pd


ROOT_PATH = "./data"

def convert_to_CSV(filename):
    try:
        print(f"File: {filename}")

        if (not os.path.exists(filename)):
            print("!!!! File not Found. !!!!")
            return

        if (filename[-4:]=='arff'):
            data, label = load_from_arff_to_dataframe(filename)
        else:
            if (not os.path.exists(filename[:-2]+'arff')):
                data, label = load_from_tsfile(filename, return_data_type='nested_univ')
            else:
                print("!! .arff file exists... Skip...")
                return           

        flattended_rows = []
        for i in range(len(data)):
            combined_series = pd.concat([data.iloc[i, j] for j in range(data.shape[1])],
                                        axis=0, ignore_index=True)
            flattended_rows.append(combined_series)
        
        df_flat = pd.DataFrame(flattended_rows)
        df_flat.columns = range(df_flat.shape[1])
        df_flat.insert(0, 'activity', label)


        name = f"{ds}_{tp}.csv"
        fullpath = f"{os.path.join(ROOT_PATH, name)}"
        df_flat.to_csv(fullpath, index=False)
        print(f"- Complete: {fullpath}")
    except Exception as e:
        print(f"- ERROR: {e}\n")

def merge_CSV(dataset, trainCSV, testCSV):

    print(f"\n dataset: {dataset} | train: {trainCSV} | test: {testCSV}\n")
    df_train = pd.read_csv(trainCSV)
    df_test = pd.read_csv(testCSV)

    label_column_name = df_train.select_dtypes(include=['object']).columns[0]

    labels_train = df_train[label_column_name]
    labels_test = df_test[label_column_name]

    all_labels = pd.concat([labels_train, labels_test], ignore_index=True)
    all_labels.to_csv(f"{ROOT_PATH}/{dataset}_labels.csv", index=False, header=[label_column_name])

    df_train = df_train.drop(columns=[label_column_name])
    df_test = df_test.drop(columns=[label_column_name])

    df_combined = pd.concat([df_train, df_test], ignore_index=True)

    date_rng = pd.date_range(start='2023-01-01', periods=len(df_combined), freq='H')
    df_combined.insert(0, 'date', date_rng)

    data_columns = {col: f'OT_{i}' for i, col in enumerate(df_combined.columns[1:])}
    df_combined = df_combined.rename(columns=data_columns)

    df_combined.to_csv(f"{ROOT_PATH}/{dataset}.csv", index=False)

for ds in ['BasicMotions', 'DuckDuckGeese', 'Epilepsy']:
    path = f"{ROOT_PATH}/{ds}"
    print(f"\n\n============ {ds} ============")
    for tp in ['TRAIN', 'TEST']:
        for extension in ['arff', 'ts']:
            convert_to_CSV(filename=f"{path}/{ds}_{tp}.{extension}")
    merge_CSV(ds, f"{ROOT_PATH}/{ds}_TRAIN.csv", f"{ROOT_PATH}/{ds}_TEST.csv")