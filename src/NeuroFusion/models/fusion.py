import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim:int, num_heads:int=8, dropout:float=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        # self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, structural: torch.Tensor, functional:torch.Tensor) )->torch.Tensor:

        # structural: (B, N_s, embed_dim) or (B, 1, embed_dim) if pooled
        # functional: (B, N_f, embed_dim) or (B, 1, embed_dim) if pooled

        if structural.dim() == 2:
            structural = structural.unsqueeze(1)  # (B, 1, embed_dim)
        if functional.dim() == 2:
            functional = functional.unsqueeze(1)  # (B, 1, embed_dim)

        B, N_q, _ = structural.size()
        B, N_kv, _ = functional.size()

        q = self.q_proj(structural)
        k = self.k_proj(functional)
        v = self.v_proj(functional)

        q = q.view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N_q, head_dim)
        k = k.view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N_kv, head_dim)
        v = v.view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N_kv, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N_q, N_kv)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N_q, self.embed_dim)  # (B, N_q, embed_dim)
        out = self.dropout(out)

        # if input was pooled, return pooled output as well 
        if structural.size(1) == 1:
            pooled_output = out.mean(dim=1, keepdim=True)
            return out, pooled_output
        else:
            return out


class GatedFusion(nn.Module):
    def __init__(self, feature_dim:int, hidden_dim:Optional[int]=None):
        super().__init__()
        hidden_dim = hidden_dim or feature_dim // 2
    
        self.gate = nn.Sequential(
            nn.Linear(feature_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid()
        )
        

    def forward(self, structural: torch.Tensor, functional: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]  :
        # structural: (B, N, embed_dim)
        # functional: (B, N, embed_dim)
        combined = torch.cat([structural, functional], dim=-1)  # (B, N, 2*embed_dim)
        gate = self.sigmoid(self.gate(combined))  # (B, N, embed_dim)
        fused = gate * structural + (1 - gate) * functional  # (B, N, embed_dim)
        return fused, gate 

class HierarchicalFusionBlock(nn.Module):
    def __init__(self, feature_dim:int, num_heads:int=8, hidden_dim:Optional[int]=None):
        super().__init__()
        self.gated_fusion = GatedFusion(feature_dim, hidden_dim)
        self.cross_attention = MultiHeadCrossAttention(feature_dim)
        self.norm = nn.LayerNorm(feature_dim)


    def forward(self, structural: torch.Tensor, functional: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # structural: (B, N, embed_dim) or (B, 1, embed_dim) if pooled
        # functional: (B, N, embed_dim) or (B, 1, embed_dim) if pooled
        
        

        # First apply gated fusion
        fused, gate = self.gated_fusion(structural, functional)
        # Then apply cross attention
        attended, pooled_output = self.cross_attention(fused, fused)
        return attended, pooled_output       