# visualize_clusters.py

import torch
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, SpectralClustering
import matplotlib.pyplot as plt
import os
import argparse

# --- PatchTST 프로젝트의 필요 모듈 가져오기 ---
from models.PatchTST import Model as PatchTST
from data_provider.data_factory import data_provider

def visualize_clustering(args):
    """
    훈련된 PatchTST 모델을 사용하여 데이터의 임베딩을 추출하고,
    t-SNE와 K-Means를 사용해 클러스터링 결과를 시각화합니다.
    """
    print("--- 1. 장치 설정 및 모델 불러오기 ---")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 훈련 시 사용했던 모델 설정값을 그대로 사용해야 합니다.
    model = PatchTST(args).float().to(device)
    
    # 훈련된 모델 가중치 불러오기
    checkpoint_path = os.path.join(args.checkpoints, args.setting, 'checkpoint.pth')
    if not os.path.exists(checkpoint_path):
        print(f"오류: 체크포인트 파일을 찾을 수 없습니다! 경로: {checkpoint_path}")
        return
        
    print(f"체크포인트 로딩: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval() # 평가 모드로 설정

    print("\n--- 2. 데이터셋 불러오기 ---")
    _, data_loader = data_provider(args, flag='test') # 테스트 데이터셋 사용

    all_embeddings = []
    print("\n--- 3. 데이터에서 임베딩(특징) 추출 중 ---")
    with torch.no_grad():
        for i, (batch_x, _, _, _) in enumerate(data_loader):
            batch_x = batch_x.float().to(device)
            
            # PatchTST 모델의 인코더를 통해 임베딩을 추출합니다.
            # 모델 내부 구조에 따라 이 부분은 수정이 필요할 수 있습니다.
            # 여기서는 인코더의 출력을 평균내어 사용합니다.
            outputs = model.forward(batch_x) # 모델마다 forward 인자가 다를 수 있음
            embeddings = outputs.detach().cpu().numpy()
            
            # (batch_size, pred_len, c_out) -> (batch_size, features)
            # 시계열 전체의 대표 임베딩을 얻기 위해 평균을 냅니다.
            avg_embedding = np.mean(embeddings, axis=1)
            all_embeddings.append(avg_embedding)
            if i > 50: # 시각화를 위해 너무 많은 데이터는 사용하지 않도록 제한
                print("효율적인 시각화를 위해 50배치만 사용합니다.")
                break

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"총 {all_embeddings.shape[0]}개의 데이터 포인트에 대한 임베딩 추출 완료.")

    print("\n--- 4. t-SNE로 차원 축소 중 (시간이 다소 걸릴 수 있습니다) ---")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, all_embeddings.shape[0]-1))
    embeddings_2d = tsne.fit_transform(all_embeddings)

    # print("\n--- 5. K-Means 클러스터링 수행 ---")
    # means = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
    # cluster_labels = kmeans.fit_predict(embeddings_2d)

    print("\n--- 5. Spectral Clustering 수행 ---")
    spectral = SpectralClustering(
        n_clusters=args.n_clusters,
        random_state=42,
        affinity='nearest_neighbors', # t-SNE 결과에는 이 옵션이 잘 작동합니다.
        n_init=10
    )
    cluster_labels = spectral.fit_predict(embeddings_2d)

    print("\n--- 6. 결과 시각화 및 저장 ---")
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
    plt.title(f'PatchTST Embeddings Sprctral Clustering (k={args.n_clusters})', fontsize=16)
    plt.xlabel('T-SNE Component 1')
    plt.ylabel('T-SNE Component 2')
    plt.legend(handles=scatter.legend_elements()[0], labels=[f'Cluster {i}' for i in range(args.n_clusters)])
    plt.grid(True)
    
    save_path = 'spectral_clustering_visualization.png'
    plt.savefig(save_path)
    print(f"성공! 클러스터링 시각화가 '{save_path}' 파일로 저장되었습니다.")


if __name__ == '__main__':
    # 훈련 스크립트와 동일한 인자들을 설정해야 합니다.
    parser = argparse.ArgumentParser(description='PatchTST Clustering Visualization')

    # 기본 설정
    parser.add_argument('--model', type=str, required=True, default='PatchTST', help='model name')
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='/home/intern/TimeCMA/dataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--n_clusters', type=int, default=7, help='number of clusters for K-Means')

    # 모델 세부 설정 (훈련 시 사용했던 값과 동일하게 유지)
    parser.add_argument('--features', type=str, default='M', help='forecasting task')
    parser.add_argument('--target', type=str, default='OT', help='target feature')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=16, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    
    # PatchTST 특화 설정
    parser.add_argument('--fc_dropout', type=float, default=0.05)
    parser.add_argument('--head_dropout', type=float, default=0.0)
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--padding_patch', default='end')
    parser.add_argument('--revin', type=int, default=1)
    parser.add_argument('--affine', type=int, default=0)
    parser.add_argument('--subtract_last', type=int, default=0)
    parser.add_argument('--decomposition', type=int, default=0)
    parser.add_argument('--kernel_size', type=int, default=25)
    parser.add_argument('--individual', type=int, default=0)
    
    # 기타
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')

    args = parser.parse_args()
    
    # 훈련 결과가 저장된 폴더 이름을 설정합니다.
    # args.setting = f"{args.data}_sl{args.seq_len}_pl{args.pred_len}_{args.model}_ft{args.features}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}"
    
    # 위 setting 형식이 실제와 다를 경우, 아래처럼 직접 지정해주세요.
    args.setting = 'ETTh1_96_96_PatchTST_ETTh1_ftM_sl96_ll48_pl96_dm128_nh16_el3_dl1_df256_fc3_ebtimeF_dtTrue_test_0'


    visualize_clustering(args)
