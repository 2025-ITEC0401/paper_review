# ./src/train.py

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import math
import numpy as np
import argparse
from tqdm import tqdm
import os
import gc # --- 가비지 컬렉션 임포트 ---

from src.encoder import TCNEncoder
from src.agents import RetrievalAgent, AugmentationAgent, ReviewAgent
from src.utils import apply_augmentation, InfoNCELoss
from src.data_loader import load_and_preprocess_data
# --- 추가된 임포트 ---
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# --------------------


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Warmup 후 Cosine 형태로 Learning Rate를 감소시키는 스케줄러를 생성합니다.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def main(args):
    # --- 1. 초기 설정 ---
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. 데이터 로딩 (CPU에 유지) ---
    full_data_tensor, labels = load_and_preprocess_data(
        dataset_name=args.dataset_name,
        root_path=args.root_path,
        seq_len=args.seq_len
    )
    if full_data_tensor is None:
        print(f"Could not load data for {args.dataset_name}. Exiting.")
        return
        
    # full_data_tensor를 CPU에 둡니다.

    # --- 3. TCN 인코더, 옵티마이저 초기화 (CPU에 유지) ---
    encoder = TCNEncoder(input_channels=full_data_tensor.shape[1], output_dim=args.repr_dim)
    optimizer = optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = None
    if args.use_scheduler:
        num_training_steps = args.epochs * min(args.bank_size, len(full_data_tensor))
        num_warmup_steps = int(num_training_steps * 0.1)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        print(f"Scheduler initialized. Total steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")

    
    # --- PHASE 1: TCN 인코더로 초기 인코딩 계산 ---
    print("\n--- PHASE 1: Calculating Initial Encodings (Encoder on GPU) ---")
    encoder.to(device) # args.gpu로 지정된 device로 보냄
    encoder.eval()
    all_encodings_list = []
    with torch.no_grad():
        for i in tqdm(range(0, len(full_data_tensor), args.batch_size), desc="Calculating initial encodings"):
            # 배치만 GPU로 이동
            batch = full_data_tensor[i:i+args.batch_size].to(device)
            # 인코딩 후 즉시 CPU로 이동
            all_encodings_list.append(encoder(batch).cpu())
            
    all_encodings = torch.cat(all_encodings_list, dim=0) # (CPU 텐서)
    
    # TCN 인코더를 CPU로 다시 이동시켜 VRAM 확보
    encoder.cpu()
    torch.cuda.empty_cache()
    print("Initial encodings calculated and Encoder moved to CPU.")


    # --- PHASE 2: LLM으로 메모리 뱅크 구축 (LLM on *Specified* GPU) ---
    print(f"\n--- PHASE 2: Building Memory Bank (LLM on {device}, Data on CPU) ---")
    print("Initializing LLM... This may take a while.")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path)
    
    # --- 💥💥💥 수정된 핵심 부분 💥💥💥 ---
    # "auto" 대신, 사용자가 --gpu 인자로 지정한 GPU에만 로드하도록 강제
    device_map = {"": device} 
    # -----------------------------------

    model = AutoModelForCausalLM.from_pretrained(
        args.llm_path,
        quantization_config=quantization_config,
        device_map=device_map # "auto" 대신 수정된 device_map 사용
    )
    model.eval()
    print(f"LLM initialized on {device}.")

    print("Initializing LLM Agents...")
    retrieval_agent = RetrievalAgent(model, tokenizer)
    augmentation_agent = AugmentationAgent(model, tokenizer)
    review_agent = ReviewAgent(model, tokenizer)
    print("LLM Agents initialized.")
    
    memory_bank = {}
    num_samples_for_bank = min(args.bank_size, len(full_data_tensor))
    sample_indices_for_bank = np.random.choice(len(full_data_tensor), num_samples_for_bank, replace=False)

    for i in tqdm(sample_indices_for_bank, desc="Building Memory Bank"):
        # 모든 데이터는 CPU에 있습니다.
        xi = full_data_tensor[i:i+1] # (CPU)
        
        # 코사인 유사도 계산 (CPU에서 수행)
        similarities = torch.nn.functional.cosine_similarity(all_encodings[i:i+1], all_encodings, dim=1)
        similarities[i] = -1
        top_k_indices = torch.topk(similarities, args.k_candidates).indices
        candidates = full_data_tensor[top_k_indices] # (CPU)
        
        # RetrievalAgent는 CPU 텐서를 받아 .tolist()로 문자열화 하므로 문제 없음
        relevant_seqs_list = retrieval_agent.get_relevant_sequences(xi.squeeze(0), candidates, m_relevant=args.m_relevant)
        
        if not relevant_seqs_list:
            continue
        # relevant_seqs도 CPU 텐서로 저장
        relevant_seqs = torch.stack([s for s in relevant_seqs_list]) # (CPU)
        
        local_approved = []
        if relevant_seqs.nelement() > 0:
            current_strategy, similar_strategy = augmentation_agent.select_strategies(xi.squeeze(0), relevant_seqs[0])
            
            x_aug_current = apply_augmentation(xi, current_strategy) # (CPU)
            if review_agent.evaluate(xi.squeeze(0), x_aug_current.squeeze(0)) == 'Correct':
                local_approved.append(x_aug_current)

            x_aug_similar = apply_augmentation(relevant_seqs[0:1], similar_strategy) # (CPU)
            if review_agent.evaluate(relevant_seqs[0], x_aug_similar.squeeze(0)) == 'Correct':
                local_approved.append(x_aug_similar)

        memory_bank[i] = {
            'relevant': relevant_seqs, # (CPU 텐서)
            'augmented': local_approved # (CPU 텐서)
        }
    
    print(f"Memory bank built for {len(memory_bank)} samples.")

    # --- PHASE 3: LLM 삭제 및 VRAM 확보 ---
    print("\n--- PHASE 3: Clearing LLM from VRAM ---")
    del retrieval_agent, augmentation_agent, review_agent, model, tokenizer
    gc.collect() # 가비지 컬렉션
    torch.cuda.empty_cache() # VRAM 캐시 비우기
    print("LLM deleted and VRAM cleared.")

    
    # --- PHASE 4: TCN 인코더 및 데이터 GPU로 이동 후 학습 ---
    print(f"\n--- PHASE 4: Starting Representation Learning (Encoder/Data on {device}) ---")
    encoder.to(device) # TCN 인코더를 다시 device로
    encoder.train()
    full_data_tensor = full_data_tensor.to(device) # 전체 데이터를 device로
    all_encodings = all_encodings.to(device) # 전체 인코딩을 device로
    print("Encoder and all data moved to GPU.")

    criterion = InfoNCELoss(temperature=args.temperature)
    
    for epoch in range(args.epochs):
        total_loss = 0
        train_indices = torch.tensor(list(memory_bank.keys()), device=device)
        train_indices = train_indices[torch.randperm(len(train_indices))]
        
        pbar = tqdm(train_indices, desc=f"Epoch {epoch+1}/{args.epochs}")
        for i_val in pbar:
            i = i_val.item()
            optimizer.zero_grad()
            
            # anchor_encoding은 full_data_tensor (GPU)에서 가져옴
            anchor_encoding = encoder(full_data_tensor[i:i+1])
            
            # memory_bank의 텐서들(CPU)을 GPU로 이동
            positives_list = [mem.to(device) for mem in memory_bank[i]['relevant']]
            for aug in memory_bank[i]['augmented']:
                positives_list.append(aug.squeeze(0).to(device))

            if not positives_list:
                continue
            
            positives = torch.stack(positives_list) # (GPU)
            positive_encodings = encoder(positives) # (GPU)
            
            all_indices = set(range(len(full_data_tensor)))
            all_indices.remove(i)
            negative_indices = torch.tensor(list(all_indices), device=device)
            negatives_encodings = all_encodings[negative_indices] # (GPU)
            
            batch_loss = 0
            for pos_encoding in positive_encodings:
                batch_loss += criterion(anchor_encoding, pos_encoding.unsqueeze(0), negatives_encodings)
            
            if positive_encodings.size(0) > 0:
                loss = batch_loss / positive_encodings.size(0)
            else:
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            if args.use_scheduler:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4e}", lr=f"{current_lr:.4e}")

        avg_loss = total_loss / len(train_indices) if len(train_indices) > 0 else 0
        print(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.4f}")
        
        # 매 에폭마다 all_encodings 업데이트 (모든 텐서가 GPU에 있으므로 그대로 수행)
        encoder.eval()
        with torch.no_grad():
            all_encodings = []
            for i in tqdm(range(0, len(full_data_tensor), args.batch_size), desc=f"Updating all encodings after Epoch {epoch+1}"):
                batch = full_data_tensor[i:i+args.batch_size]
                all_encodings.append(encoder(batch))
            all_encodings = torch.cat(all_encodings, dim=0)
        encoder.train()

    # --- 6. 결과 저장 ---
    print("\nTraining finished. Saving model, representations, and labels...")
    os.makedirs('./saved_models', exist_ok=True)
    os.makedirs('./saved_reps', exist_ok=True)

    model_save_path = f'./saved_models/{args.dataset_name}_encoder.pth'
    reps_save_path = f'./saved_reps/{args.dataset_name}_representations.npy'
    labels_save_path = f'./saved_reps/{args.dataset_name}_labels.npy'

    torch.save(encoder.state_dict(), model_save_path)
    final_encodings_cpu = all_encodings.cpu().numpy()
    np.save(reps_save_path, final_encodings_cpu)

    if labels is not None:
        np.save(labels_save_path, labels)

    print(f"✅ Encoder saved to {model_save_path}")
    print(f"✅ Representations saved to {reps_save_path}")
    if labels is not None:
        print(f"✅ Labels saved to {labels_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MERIT Training Script")
    parser.add_argument('--dataset_name', type=str, default='BasicMotions', help='Dataset name (e.g., ETTh1, BasicMotions)')
    parser.add_argument('--root_path', type=str, default='./datasets/', help='Root path of the dataset')
    parser.add_argument('--llm_path', type=str, default='./models/llama-3.1-8b-instruct', help='Path to local LLM model')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id')
    
    parser.add_argument('--seq_len', type=int, default=96, help='Input sequence length (ignored for UEA datasets)')
    parser.add_argument('--repr_dim', type=int, default=320, help='Representation dimension')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs') # 논문 값 100
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for encoding') # 논문 값 128

    # --- 수정된 부분: 논문 값으로 기본값 변경 ---
    parser.add_argument('--lr', type=float, default=1e-3, help='Peak learning rate (Paper: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (Paper: 0.0005)')
    # ---------------------------------------------
    
    parser.add_argument('--bank_size', type=int, default=100, help='Number of samples to build memory bank for')
    
    # --- 수정된 부분: 논문 값으로 기본값 변경 ---
    parser.add_argument('--k_candidates', type=int, default=5, help='Number of initial candidates for retrieval (Paper: K=5)')
    parser.add_argument('--m_relevant', type=int, default=3, help='Number of relevant sequences to select (Paper: M=3)')
    # ---------------------------------------------
    
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for contrastive loss')

    # --- 추가된 부분: 스케줄러 사용 여부 ---
    parser.add_argument('--use_scheduler', action='store_true', help='Use cosine warmup scheduler')
    # ------------------------------------
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')

    args = parser.parse_args()
    main(args)