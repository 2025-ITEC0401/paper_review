import pandas as pd
import re
import os

path = f'./'
list = os.listdir(path)
list = [f for f in list if f.endswith(".log")]
df1 = pd.DataFrame({})
df2 = pd.DataFrame({})

def parsing(targetFile, index_label, type):
    elapsed_list = []
    file = open(targetFile, 'r', encoding='utf-8')
    epoch = 0
    for line in file:
        if (type == "T"):
            match = re.search(r'Epoch: (\d+), Train Loss: (\d+\.\d+)', line)
        else:
            match = re.search(r'Epoch: (\d+), Valid Loss: (\d+\.\d+)', line)
        outlen = re.search(r"(?<=_)o\d+", targetFile).group(0)

        if match:
            epoch = match.group(1)
            elapsed_list.append(match.group(2))
            index_label.append(epoch)
    df_tmp = pd.DataFrame({
        outlen: elapsed_list
    })
    
    return outlen, df_tmp
    
excel=pd.ExcelWriter(path + 'res.xlsx', engine='openpyxl')

for file in list:
    indexList = []
    col, val = parsing(path + file, indexList, "T")
    df1[col] = val

df1.index = indexList
df1.to_excel(excel, sheet_name="Train", index=True)

for file in list:
    indexList = []
    col, val = parsing(path + file, indexList, "V")
    df2[col] = val

df2.index = indexList
df2.to_excel(excel, sheet_name="Valid", index=True)
excel.close()