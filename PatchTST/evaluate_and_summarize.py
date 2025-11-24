# evaluate_and_summarize.py

import argparse
import pandas as pd
from evaluate_clusters import evaluate # 수정된 evaluate 함수 import

# 1. 평가할 데이터셋 및 설정 정의
datasets = {
    "BasicMotions": {"seq_len": 100, "enc_in": 6, "patch_len": 10, "stride": 5},
    "Epilepsy": {"seq_len": 206, "enc_in": 3, "patch_len": 16, "stride": 8},
    "Libras": {"seq_len": 45, "enc_in": 2, "patch_len": 8, "stride": 4},
    "HandMovementDirection": {"seq_len": 400, "enc_in": 10, "patch_len": 16, "stride": 8},
}

# 2. 공통 인자 설정 (evaluate_clusters.py와 동일하게 유지)
def get_common_args(gpu_id, root_path):
    parser = argparse.ArgumentParser()
    # 기본 경로 및 GPU 설정
    parser.add_argument('--root_path', type=str, default=root_path)
    parser.add_argument('--gpu', type=int, default=gpu_id)
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    # 모델 구조 인자 (훈련 시와 동일하게)
    parser.add_argument('--pred_len', type=int, default=24)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=16)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.2)
    # PatchTST 기본 설정값들
    parser.add_argument('--task_name', type=str, default='reconstruction')
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--fc_dropout', type=float, default=0.05)
    parser.add_argument('--head_dropout', type=float, default=0.0)
    parser.add_argument('--padding_patch', default='end')
    parser.add_argument('--revin', type=int, default=0)
    parser.add_argument('--affine', type=int, default=0)
    parser.add_argument('--subtract_last', type=int, default=0)
    parser.add_argument('--decomposition', type=int, default=0)
    parser.add_argument('--kernel_size', type=int, default=25)
    # parser.add_argument('--individual', type=int, default=0) # 삭제됨
    parser.add_argument('--embed_type', type=int, default=0)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--output_attention', action='store_true')
    parser.add_argument('--c_out', type=int)

    # 실제 사용되지 않는 인자지만 에러 방지 위해 임시값 할당
    args, _ = parser.parse_known_args()
    return args

# 3. 메인 실행 로직
if __name__ == "__main__":
    
    # 터미널에서 GPU ID와 데이터 경로 입력받기
    main_parser = argparse.ArgumentParser(description='Evaluate multiple datasets and summarize results.')
    main_parser.add_argument('--gpu', type=int, default=1, help='GPU ID to use.')
    main_parser.add_argument('--root_path', type=str, default="/hdd/dataset/newDataset", help='Root path for datasets.')
    main_args = main_parser.parse_args()

    all_results = {}
    common_args = get_common_args(main_args.gpu, main_args.root_path)

    # 각 데이터셋에 대해 평가 실행 및 결과 저장
    for name, config in datasets.items():
        print(f"\n===== Evaluating {name} =====")
        # 데이터셋별 특화 인자 업데이트
        args = argparse.Namespace(**vars(common_args)) # 공통 인자 복사
        args.dataset_name = name
        args.seq_len = config["seq_len"]
        args.enc_in = config["enc_in"]
        args.patch_len = config["patch_len"]
        args.stride = config["stride"]
        args.c_out = args.enc_in # 중요: c_out 설정

        # 평가 함수 호출
        result = evaluate(args)
        if result:
            all_results[name] = result

    # 4. 결과 표 생성 및 출력
    if all_results:
        print("\n\n===== 최종 결과 요약표 =====")
        df = pd.DataFrame.from_dict(all_results, orient='index')
        df = df.round(4) # 소수점 4자리까지 표시

        # Markdown 형식으로 출력
        print(df.to_markdown())

        # (선택 사항) CSV 파일로 저장
        # df.to_csv("clustering_summary.csv")
        # print("\n결과가 clustering_summary.csv 파일로도 저장되었습니다.")
    else:
        print("\n오류: 평가 결과가 없습니다.")
