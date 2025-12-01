from sktime.datasets import load_UCR_UEA_dataset
import pandas as pd

print("데이터셋 로딩을 시작합니다...")

try:
    # 1. BasicMotions 불러오기
    X_bm, y_bm = load_UCR_UEA_dataset(name="BasicMotions", return_X_y=True)
    print(f"✅ BasicMotions 로딩 성공! 데이터 형태: {X_bm.shape}")

    # 2. DuckDuckGeese 불러오기
    X_ddg, y_ddg = load_UCR_UEA_dataset(name="DuckDuckGeese", return_X_y=True)
    print(f"✅ DuckDuckGeese 로딩 성공! 데이터 형태: {X_ddg.shape}")

    # 3. Epilepsy 불러오기
    X_ep, y_ep = load_UCR_UEA_dataset(name="Epilepsy", return_X_y=True)
    print(f"✅ Epilepsy 로딩 성공! 데이터 형태: {X_ep.shape}")

    print("\n모든 데이터셋을 성공적으로 메모리에 불러왔습니다.")
    # 참고: 다운로드된 실제 파일은 홈 디렉토리의 '~/sktime_data/' 폴더에 저장됩니다.

except Exception as e:
    print(f"오류 발생: {e}")
    print("인터넷 연결을 확인하거나, sktime 라이브러리가 최신 버전인지 확인해 보세요.")
