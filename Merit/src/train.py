# src/train.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
from tqdm import tqdm

from src.encoder import TCNEncoder
from src.agents import RetrievalAgent, AugmentationAgent, ReviewAgent
from src.utils import apply_augmentation, InfoNCELoss

def main(args):
    # --- 1. 초기 설정 ---
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. 데이터 로딩 (예시) ---
    # 실제로는 args.dataset_path를 이용해 UEA/UCR 데이터셋을 로드해야 합니다.
    # 여기서는 더미 데이터로 형식을 보여줍니다.
    # 데이터는 (샘플 수, 채널 수, 시계열 길이) 형태여야 합니다.
    dummy_data = torch.randn(100, 1, 128) 
    dataset = TensorDataset(dummy_data)
    # 전체 데이터를 메모리에 올리고 사용
    full_data_tensor = next(iter(DataLoader(dataset, batch_size=len(dataset))))[0].to(device)

    # --- 3. 모델 및 Agent 초기화 ---
    encoder = TCNEncoder(input_channels=full_data_tensor.shape[1]).to(device)
    optimizer = optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=1e-5)
    
    print("Initializing LLM Agents... This may take a while.")
    retrieval_agent = RetrievalAgent(args.llm_path, device)
    augmentation_agent = AugmentationAgent(args.llm_path, device)
    review_agent = ReviewAgent(args.llm_path, device)
    print("LLM Agents initialized.")

    # --- 4. 메모리 뱅크 구축 (Algorithm 1의 1-29행) ---
    print("Building memory bank with LLM agents...")
    memory_bank = {}
    
    # 시계열 인코딩 미리 계산
    with torch.no_grad():
        all_encodings = encoder(full_data_tensor)

    for i in tqdm(range(len(full_data_tensor)), desc="Building Memory Bank"):
        xi = full_data_tensor[i:i+1]
        
        # 1. Retrieval Agent - Step 1: Candidate Selection (Top-K)
        similarities = torch.nn.functional.cosine_similarity(all_encodings[i:i+1], all_encodings)
        similarities[i] = -1 # 자기 자신 제외
        top_k_indices = torch.topk(similarities, args.k_candidates).indices
        candidates = full_data_tensor[top_k_indices]
        
        # 2. Retrieval Agent - Step 2: LLM Refinement (Top-M)
        # 논문에서는 M=3이지만, 여기서는 편의상 1개만 선택
        relevant_seqs = retrieval_agent.get_relevant_sequences(xi.squeeze(), candidates.squeeze(1))
        
        # 3. Augmentation & Review Loop
        # 실제 구현에서는 루프가 필요하지만, 여기서는 1회만 실행하는 것으로 간소화
        local_approved = []
        
        # 원본 시퀀스에 대한 증강
        current_strategy, _ = augmentation_agent.select_strategies(xi.squeeze(), relevant_seqs[0])
        x_aug_current = apply_augmentation(xi, current_strategy)
        if review_agent.evaluate(xi.squeeze(), x_aug_current.squeeze()) == 'Correct':
            local_approved.append(x_aug_current)

        # 유사 시퀀스에 대한 증강
        _, similar_strategy = augmentation_agent.select_strategies(xi.squeeze(), relevant_seqs[0])
        x_aug_similar = apply_augmentation(torch.tensor(relevant_seqs, device=device).unsqueeze(0), similar_strategy)
        if review_agent.evaluate(torch.tensor(relevant_seqs, device=device), x_aug_similar.squeeze()) == 'Correct':
            local_approved.append(x_aug_similar)

        memory_bank[i] = {
            'relevant': torch.tensor(relevant_seqs, device=device), 
            'augmented': local_approved
        }
    
    # --- 5. 시계열 표현 학습 (Algorithm 1의 30-41행) ---
    print("\nStarting Time Series Representation Learning...")
    criterion = InfoNCELoss(temperature=args.temperature)
    
    for epoch in range(args.epochs):
        total_loss = 0
        
        indices = torch.randperm(len(full_data_tensor)) # 매 에포크마다 순서 섞기
        
        for i in tqdm(indices, desc=f"Epoch {epoch+1}/{args.epochs}"):
            i = i.item()
            optimizer.zero_grad()
            
            anchor_encoding = encoder(full_data_tensor[i:i+1])
            
            # Positive 샘플 구성
            positives = memory_bank[i]['relevant']
            for aug in memory_bank[i]['augmented']:
                positives = torch.cat((positives, aug), dim=0)

            if len(positives) == 0:
                continue
            
            # 무작위로 Positive 샘플 하나 선택
            positive_sample = positives[torch.randint(len(positives), (1,))].unsqueeze(0)
            positive_encoding = encoder(positive_sample)
            
            # Negative 샘플 구성 (현재 샘플과 Positive 제외)
            all_indices = set(range(len(full_data_tensor)))
            all_indices.remove(i)
            # (시간 관계상 Positive 인덱스는 제외하지 않음, 영향 미미)
            negative_indices = torch.tensor(list(all_indices), device=device)
            negatives_encodings = all_encodings[negative_indices]
            
            # Loss 계산 및 역전파
            loss = criterion(anchor_encoding, positive_encoding, negatives_encodings)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {total_loss / len(full_data_tensor):.4f}")
        
        # 에포크마다 인코더 재계산 (선택적이지만 성능 향상에 도움)
        with torch.no_grad():
            all_encodings = encoder(full_data_tensor)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MERIT Training Script")
    parser.add_argument('--dataset_path', type=str, default='./data/dummy', help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')

    parser.add_argument('--llm_path', type=str, default='./models/llama-3.1-8b-instruct', help='Path to local LLM model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--k_candidates', type=int, default=5, help='Number of initial candidates for retrieval')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for contrastive loss')

    args = parser.parse_args()
    main(args)
