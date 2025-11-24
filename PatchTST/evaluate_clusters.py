# evaluate_clusters.py (결과 반환 기능 추가)

import argparse
import torch
import numpy as np
import os
import matplotlib.pyplot as plt

# sktime 로더 및 클러스터링/평가 모듈
from sktime.datasets._data_io import load_from_tsfile
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import normalized_mutual_info_score as NMI, rand_score as RI
from sklearn.preprocessing import LabelEncoder

# 훈련 스크립트에서 모델 클래스 가져오기
from models.PatchTST import Model as PatchTST

# --- [수정] 함수가 점수를 반환하도록 변경 ---
def evaluate(args):
    print(f"\n--- {args.dataset_name} 평가 시작 ---")

    # 0. 장치 설정
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # 1. 테스트 데이터셋 로드
    print("... 테스트 데이터 로딩 중 ...")
    test_file = os.path.join(args.root_path, args.dataset_name, f"{args.dataset_name}_TEST.ts")
    X_test_pd, y_test = load_from_tsfile(test_file)
    print(test_file)
    print(f"... 테스트 샘플 수: {X_test_pd.shape[0]} ...")

    # 레이블(y)을 0부터 시작하는 정수로 변환
    le = LabelEncoder()
    y_true = le.fit_transform(y_test)
    n_clusters = len(np.unique(y_true))
    print(f"... 실제 클래스(군집) 개수: {n_clusters} ...")

    # --- sktime(nested pandas)를 3D Numpy로 변환 ---
    def convert_to_3d_numpy(X_pd):
        n_samples = X_pd.shape[0]
        n_channels = X_pd.shape[1]
        n_timesteps = X_pd.iloc[0, 0].shape[0]
        arr = np.empty((n_samples, n_timesteps, n_channels), dtype=np.float32)
        for i in range(n_samples):
            for j in range(n_channels):
                arr[i, :, j] = X_pd.iloc[i, j].to_numpy()
        return arr

    print("... 3D Numpy 배열로 변환 중 ...")
    X_test_np = convert_to_3d_numpy(X_test_pd)
    print("... 변환 완료 ...")
    # --- [변환 완료] ---

    X_test_tensor = torch.from_numpy(X_test_np).float().to(device) # (B, L, C)

    # 2. 훈련된 모델 로드
    print("... 훈련된 모델 로딩 중 ...")
    args.individual = 0 # 모델 로딩 시 필요한 인자 추가
    model = PatchTST(args).float().to(device)
    setting = f'{args.dataset_name}_sl{args.seq_len}_pl{args.pred_len}_dm{args.d_model}'
    checkpoint_path = os.path.join(args.checkpoints, setting, 'checkpoint.pth')
    
    # 체크포인트 파일 존재 확인
    if not os.path.exists(checkpoint_path):
        print(f"오류: 체크포인트 파일을 찾을 수 없습니다! 경로: {checkpoint_path}")
        print(f"--- {args.dataset_name} 평가 건너뜀 ---")
        return None # <-- 파일 없으면 None 반환
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True)) # weights_only=True 추가 (경고 제거)
    model.eval()

    # 3. 임베딩 추출
    print("... 임베딩 추출 중 ...")
    with torch.no_grad():
        outputs = model(X_test_tensor)
    embeddings = outputs.mean(dim=1).cpu().numpy() # (B, C)
    print(f"... 추출된 임베딩 형태: {embeddings.shape} ...")

    # 4. t-SNE (시각화용)
    print("... t-SNE 차원 축소 중 ...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_2d = tsne.fit_transform(embeddings)

    # 5. K-Means 클러스터링 및 평가
    print("... K-Means 클러스터링 및 평가 ...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_preds = kmeans.fit_predict(embeddings)
    kmeans_ri = RI(y_true, kmeans_preds)
    kmeans_nmi = NMI(y_true, kmeans_preds)
    print(f"  [K-Means] RI: {kmeans_ri:.4f}, NMI: {kmeans_nmi:.4f}") # 터미널에도 출력

    # 6. Spectral 클러스터링 및 평가
    print("... Spectral 클러스터링 및 평가 ...")
    spectral = SpectralClustering(n_clusters=n_clusters, random_state=42, affinity='nearest_neighbors', n_init=10)
    spectral_preds = spectral.fit_predict(embeddings) # 원본 임베딩 사용
    spectral_ri = RI(y_true, spectral_preds)
    spectral_nmi = NMI(y_true, spectral_preds)
    print(f"  [Spectral] RI: {spectral_ri:.4f}, NMI: {spectral_nmi:.4f}") # 터미널에도 출력

    # 7. 시각화 저장
    print("... 시각화 파일 저장 중 ...")
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    # ... (시각화 코드 동일) ...
    axes[0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=y_true, cmap='viridis', alpha=0.7)
    axes[0].set_title(f'Ground Truth') # 한글 제거
    axes[1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=kmeans_preds, cmap='viridis', alpha=0.7)
    axes[1].set_title(f'K-Means (RI: {kmeans_ri:.3f}, NMI: {kmeans_nmi:.3f})')
    axes[2].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=spectral_preds, cmap='viridis', alpha=0.7)
    axes[2].set_title(f'Spectral (RI: {spectral_ri:.3f}, NMI: {spectral_nmi:.3f})')
    fig.suptitle(f'{args.dataset_name} - PatchTST Embedding Clustering Results', fontsize=16) # 한글 제거
    save_path = f'./{args.dataset_name}_clustering_results.png'
    plt.savefig(save_path)
    plt.close(fig) # 메모리 누수 방지
    print(f"--- {args.dataset_name} 평가 완료. 결과가 {save_path} 에 저장되었습니다. ---")

    # --- [수정] 결과를 딕셔너리로 반환 ---
    return {
        "KMeans_RI": kmeans_ri,
        "KMeans_NMI": kmeans_nmi,
        "Spectral_RI": spectral_ri,
        "Spectral_NMI": spectral_nmi
    }
# --- [수정 완료] ---

if __name__ == '__main__':
    # (이 부분은 직접 실행되지 않으므로 수정 불필요)
    # ... (기존 parser 설정 코드) ...
    parser = argparse.ArgumentParser(description='PatchTST Clustering Evaluation')
    # ... (기존 parser 설정 코드) ...
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--root_path', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seq_len', type=int, required=True)
    parser.add_argument('--enc_in', type=int, required=True)
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--pred_len', type=int, default=24)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=16)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
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

    args = parser.parse_args()
    args.c_out = args.enc_in

    # 직접 실행 시 결과만 출력 (표 생성 X)
    results = evaluate(args)
    if results:
         print("\n--- 최종 결과 ---")
         print(results)
