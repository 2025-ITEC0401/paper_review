# models/PatchTST.py (새 Backbone 및 Reconstruction Head 적용)

import torch
import torch.nn as nn
from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp

class Model(nn.Module):
    """
    PatchTST
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # --- Backbone 초기화 (사용자 제공 backbone 기준) ---
        self.model = PatchTST_backbone(
            c_in=configs.enc_in, 
            context_window=configs.seq_len, 
            target_window=configs.pred_len, # target_window는 backbone에서 사용 안 할 수 있음
            patch_len=configs.patch_len,    
            stride=configs.stride,          
            # max_seq_len=configs.seq_len + configs.pred_len, # 사용자 backbone에 없음
            n_layers=configs.e_layers,
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            d_k=configs.d_model // configs.n_heads if configs.n_heads > 0 else None, 
            d_v=configs.d_model // configs.n_heads if configs.n_heads > 0 else None, 
            d_ff=configs.d_ff,
            # norm='BatchNorm', # 사용자 backbone에 없음
            attn_dropout=configs.dropout, # backbone은 attn_dropout 인자 없음 -> dropout 사용
            dropout=configs.dropout,
            act=configs.activation,
            # key_padding_mask='auto', # 사용자 backbone에 없음
            # padding_var=None, # 사용자 backbone에 없음
            # attn_mask=None, # 사용자 backbone에 없음
            res_attention=True, # 사용자 backbone에 없음
            pre_norm=False, # 사용자 backbone에 없음
            store_attn=False, # 사용자 backbone에 없음
            # pe='zeros', # 사용자 backbone에 없음
            # learn_pe=True, # 사용자 backbone에 없음
            # fc_dropout=configs.fc_dropout, # 사용자 backbone에 없음
            # head_dropout=configs.head_dropout, # 사용자 backbone에 없음
            # padding_patch=configs.padding_patch, # 사용자 backbone에 없음
            # pretrain_head=False, # 사용자 backbone에 없음
            # head_type='flatten', # 사용자 backbone에 없음
            # individual=configs.individual, # 사용자 backbone에 없음 (항상 individual 처리)
            revin=configs.revin,
            affine=configs.affine,
            subtract_last=configs.subtract_last,
            verbose=False, 
            # output_dim=configs.c_out # 사용자 backbone에 없음
        )
        # --- Backbone 초기화 완료 ---
        
        self.decomposition = configs.decomposition
        if self.decomposition:
            self.decomposition_layer = series_decomp(configs.kernel_size)
            
        # --- Reconstruction Head 추가 ---
        # Backbone 출력 (B, C, D, P) -> (B, C, D*P) -> Linear -> (B, C, L)
        self.head_nf = configs.d_model * self.model.patch_num # d_model * patch_num
        self.head = nn.Linear(self.head_nf, self.seq_len)
        # --- Head 추가 완료 ---

    def reconstruction(self, x_enc):
        # Normalization (Revin 없을 시)
        # 중요: 사용자 backbone은 (B, C, L) 입력을 기대
        if self.model.revin == 0:
            x_enc = x_enc.permute(0, 2, 1) # (B, L, C) -> (B, C, L)
            means = x_enc.mean(2, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=2, keepdim=True, unbiased=False) + 1e-5).detach()
            x_enc /= stdev
        else:
             # RevIN 사용 시에도 Backbone 입력을 위해 permute 필요
             x_enc = x_enc.permute(0, 2, 1) # (B, L, C) -> (B, C, L)


        # Backbone forward: Input (B, C, L) -> Output (B, C, D, P)
        z = self.model(x_enc) 
        
        # Reshape and Project using Head: (B, C, D, P) -> (B, C, L)
        z = z.permute(0, 1, 3, 2) # (B, C, D, P) -> (B, C, P, D)
        z = torch.reshape(z, (z.shape[0], z.shape[1], -1)) # (B, C, P, D) -> (B, C, P*D)
        outputs = self.head(z) # (B, C, P*D) -> (B, C, L)
        
        # De-Normalization
        if self.model.revin == 0:
            outputs = outputs * \
                      (stdev.repeat(1, 1, self.seq_len))
            outputs = outputs + \
                      (means.repeat(1, 1, self.seq_len))
        
        # 최종 출력을 (B, L, C) 형태로 변환 (Loss 계산 위해)
        outputs = outputs.permute(0, 2, 1) # (B, C, L) -> (B, L, C)

        return outputs

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # task_name에 따라 분기 (지금은 reconstruction만 구현)
        if self.task_name == 'reconstruction':
            dec_out = self.reconstruction(x_enc)
            return dec_out
        else:
            # 다른 task (forecast, imputation 등)은 구현 필요
            raise NotImplementedError(f"Task '{self.task_name}' is not implemented for this model setup.")
            
        return None
