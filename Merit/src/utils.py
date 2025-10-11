# src/utils.py

import torch
import torch.nn.functional as F
import numpy as np

# --- 데이터 증강 함수들 ---

def jitter(x, sigma=0.03):
    # torch.Tensor를 입력으로 가정 (N, C, L)
    return x + torch.randn_like(x) * sigma

def scale(x, sigma=0.1):
    factor = torch.randn(x.shape[0], x.shape[1], 1, device=x.device) * sigma + 1.
    return x * factor

def permutation(x, max_segments=5, seg_mode="equal"):
    # ... 복잡한 증강 로직 구현 ...
    # 간단한 예시로 전체를 섞는 것으로 대체
    orig_steps = np.arange(x.shape[2])
    num_segs = np.random.randint(1, max_segments)
    
    ret = np.zeros_like(x.cpu().numpy())
    if num_segs > 1:
        if seg_mode == "random":
            split_points = np.random.choice(x.shape[2] - 2, num_segs - 1, replace=False)
            split_points.sort()
            splits = np.split(orig_steps, split_points)
        else:
            splits = np.array_split(orig_steps, num_segs)
        
        np.random.shuffle(splits)
        new_order = np.concatenate(splits)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                 ret[i,j,:] = x[i,j,new_order].cpu().numpy()
        return torch.from_numpy(ret).to(x.device)
    else:
        return x

def apply_augmentation(x, strategy_name):
    """주어진 전략 이름에 따라 증강을 적용"""
    if strategy_name.lower().strip() == 'jittering':
        return jitter(x)
    elif strategy_name.lower().strip() == 'sailing': # Scaling의 오타로 추정, 논문 참조
        return scale(x)
    elif strategy_name.lower().strip() == 'permutation':
        return permutation(x)
    # ... 다른 증강 전략들 추가 ...
    else: # 지원하지 않는 전략이면 원본 반환
        return x

# --- Contrastive Loss ---

class InfoNCELoss(torch.nn.Module):
    """InfoNCE Loss 함수 (Contrastive Loss)"""
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negatives):
        # anchor: (1, D), positive: (1, D), negatives: (M, D)
        
        # 긍정적 쌍의 유사도
        l_pos = torch.einsum('nc,nc->n', [anchor, positive]).unsqueeze(-1) # (1, 1)

        # 부정적 쌍의 유사도
        l_neg = torch.einsum('nc,mc->nm', [anchor, negatives]) # (1, M)

        # 로짓 계산
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        
        # CrossEntropyLoss 계산 (정답 레이블은 항상 0)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=anchor.device)
        return F.cross_entropy(logits, labels)
