# layers/PatchTST_backbone.py (사용자 제공 버전)
import torch
import torch.nn as nn
from layers.RevIN import RevIN

# --- 이 아래 코드는 제가 수정한 최종 버전입니다 ---

def get_activation_fn(activation):
    if activation == "relu": return nn.ReLU()
    elif activation == "gelu": return nn.GELU()
    raise ValueError(f'{activation} is not available. Please use "relu" or "gelu"')

class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, dropout=0.1, activation="gelu", n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([TSTEncoderLayer(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, dropout=dropout, activation=activation) for i in range(n_layers)])

    def forward(self, src, key_padding_mask=None, attn_mask=None):
        output = src
        scores = None
        for mod in self.layers: output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return output

class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, dropout=0.1, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = nn.Sequential(nn.LayerNorm(d_model))
        
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff),
                                 get_activation_fn(activation),
                                 nn.Dropout(dropout),
                                 nn.Linear(d_ff, d_model))
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = nn.Sequential(nn.LayerNorm(d_model))

    def forward(self, src, prev=None, key_padding_mask=None, attn_mask=None):
        src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        src = src + self.dropout_attn(src2)
        src = self.norm_attn(src)
        
        src2 = self.ff(src)
        src = src + self.dropout_ffn(src2)
        src = self.norm_ffn(src)
        
        if prev is None: scores = attn
        else: scores = prev + attn
        return src, scores

class PatchTST_backbone(nn.Module):
    def __init__(self, c_in, context_window, target_window, patch_len, stride, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu",
                 res_attention=True, pre_norm=False, store_attn=False, revin=True,
                 affine=True, subtract_last=False, verbose=False, **kwargs):
        super().__init__()
        
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        self.patch_len = patch_len
        self.stride = stride
        
        # --- START: 핵심 수정 (패치 개수 계산 로직 수정) ---
        self.padding_patch_layer = None
        if (context_window - patch_len) % stride != 0:
            padding = stride - ((context_window - patch_len) % stride)
            self.padding_patch_layer = nn.ReplicationPad1d((0, padding)) # Pad sequence dimension
            padded_context_window = context_window + padding
        else:
            padded_context_window = context_window
            
        self.patch_num = (padded_context_window - patch_len) // stride + 1
        # --- END: 핵심 수정 ---

        self.W_P = nn.Linear(patch_len, d_model)
        self.W_pos = nn.Parameter(torch.randn(self.patch_num, d_model))
        self.dropout = nn.Dropout(dropout)
        self.encoder = TSTEncoder(self.patch_num, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, dropout=dropout, activation=act, n_layers=n_layers)

    def forward(self, x) -> torch.Tensor: # Input x: [bs x nvars x seq_len]
        n_vars = x.shape[1] 
        seq_len = x.shape[2] # Original sequence length

        if self.revin:
            x = self.revin_layer(x, 'norm')

        if self.padding_patch_layer is not None:
            x = self.padding_patch_layer(x) # Pad sequence dimension
        
        # Patching
        # x: [bs x nvars x seq_len_padded]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x: [bs x nvars x patch_num x patch_len]
        
        x = self.W_P(x)
        # x: [bs x nvars x patch_num x d_model]

        u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # u: [bs * nvars x patch_num x d_model]
        
        u = self.dropout(u + self.W_pos)
        
        z = self.encoder(u)
        # z: [bs * nvars x patch_num x d_model]
        
        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))
        # z: [bs x nvars x patch_num x d_model]
        
        # Original code permuted to (B, C, D, P). Keeping this shape.
        z = z.permute(0, 1, 3, 2) # B, C, D, P 
        # z: [bs x nvars x d_model x patch_num]

        return z
