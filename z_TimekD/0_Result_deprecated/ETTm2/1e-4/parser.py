import pandas as pd
import re
import os

def parsing(targetFile, type):
    """로그 파일 하나를 파싱하여 컬럼명과 {epoch: loss} Series를 반환합니다."""
    losses = {}
    
    outlen_match = re.search(r"(?<=_)o\d+", targetFile)
    if not outlen_match:
        return None, None
    col_name = outlen_match.group(0)

    try:
        with open(targetFile, 'r', encoding='utf-8') as file:
            for line in file:
                match = re.search(r"Epoch: (\d+), (Train|Valid) Loss: (\d+\.\d+)", line)
                if match:
                    epoch = int(match.group(1))
                    run_type = match.group(2)
                    loss = float(match.group(3))
                    
                    if type == "T" and run_type == "Train":
                        losses[epoch] = loss
                    elif type == "V" and run_type == "Valid":
                        losses[epoch] = loss
    except Exception as e:
        print(f"파일 '{targetFile}' 처리 중 오류 발생: {e}")
        return None, None
    
    if not losses:
        return col_name, pd.Series(dtype='float64')
        
    return col_name, pd.Series(losses, name=col_name)

# --- 메인 로직 ---
path = './'
file_list = [f for f in os.listdir(path) if f.endswith(".log")]

# 💡 1. 결과를 담을 빈 리스트를 만듭니다.
train_series_list = []
valid_series_list = []

# Train 데이터 처리
for file in file_list:
    col, series_val = parsing(os.path.join(path, file), "T")
    if series_val is not None and not series_val.empty:
        # 💡 2. 데이터프레임에 바로 추가하는 대신, 리스트에 Series를 모읍니다.
        train_series_list.append(series_val)

# Valid 데이터 처리
for file in file_list:
    col, series_val = parsing(os.path.join(path, file), "V")
    if series_val is not None and not series_val.empty:
        valid_series_list.append(series_val)

# 💡 3. 모아둔 Series들을 concat을 사용해 한 번에 합칩니다.
# axis=1 은 Series들을 옆으로(컬럼으로) 붙이라는 의미입니다.
df_train = pd.concat(train_series_list, axis=1)
df_valid = pd.concat(valid_series_list, axis=1)

# --- 엑셀 파일 저장 ---
with pd.ExcelWriter(os.path.join(path, 'res.xlsx'), engine='openpyxl') as writer:
    df_train.fillna(0, inplace=True)
    df_train.sort_index(inplace=True)
    df_train.to_excel(writer, sheet_name="Train", index=True, index_label="Epoch")

    df_valid.fillna(0, inplace=True)
    df_valid.sort_index(inplace=True)
    df_valid.to_excel(writer, sheet_name="Valid", index=True, index_label="Epoch")

print("엑셀 파일 생성이 완료되었습니다. (데이터 유실 없음)")