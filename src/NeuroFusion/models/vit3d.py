import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, List
import math 

class PatchEmbed3D(nn.Module):
    # 3D Image to patch embedding
    def __init__(self, img_size:int = 96, patch_size:int = 16, in_chans:int=1, embed_dim:int=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size**3

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        # x: (B, C, D, H, W)
        x = self.proj(x) # (B, embed_dim, grid, grid, grid)
        x = x.flatten(2).transpose(1,2)
        return x


class Attention3D(nn.Module):
    def __init__(self, dim:int, num_heads:int=12, qkv_bias:bool=True, attn_drop:float=0.0, proj_drop=0.0):
        self.num_heads = num_heads
        self.head_dim = dim/num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q@k.transpose(-2, -1))*self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn@v).transpose(1,2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp3D(nn.Module):
    def __init__(self, in_features:int, hidden_feature:Optional[int]=None, out_features: Optional[int]=None, act_layer:nn.Module=nn.GELU, drop:float=0.0):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_feature)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_feature, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block3D(nn.Module):
    def __init__(self, dim:int, num_heads:int, mlp_ratio:float=4.0, qkv_bias:bool=True, drop:float=0.0, attn_drop:float = 0.0, act_layer: nn.Module=nn.GELU):
        super().__init__()
        self.attn = Attention3D(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp3D(in_features = dim, hidden_features =int(dim*mlp_ratio), act_layer = act_layer, drop=drop )

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer3D(nn.Module):
    def __init__(self, config: 'ModelConfig'):
        super().__init__()
        self.config = config
        self.num_features = config.embed_dim

        # patch embedding 
        self.patch_embed = PatchEmbed3D(
            img_size= config.img_size,
            patch_size = config.patch_size,
            in_chans = config.in_channels,
            embed_dim = config.embed_dim
        )

        self.num_patches = self.patch_embed.num_patches

        # positive embedding 
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, config.embed_dim))
        self.pos_drop = nn.Dropout(p=config.dropout)

        # Transformer blocks 
        self.blocks = nn.ModuleList([
            Block3D(
                dim = config.embed_dim,
                num_heads = config.num_heads,
                mlp_ratio = config.mlp_ratio,
                drop = config.dropout
            )
            for _ in range(config.num_layers)
        ])

        self.norm = nn.LayerNorm(config.embed_dim)
        self.init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights_apply)

    def _init_weights_apply(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x:torch.Tensor, return_intermediate:bool=True)->Tuple[torch.Tensor, List[torch.Tensor]]:
        # x: (B, 1, D, H, W)
        x = self.patch_embed(x) # (B, N, embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        intermediate_features = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            layer_idx = i+1
            if return_intermediate and layer_idx in self.config.structural_stages:
                intermediate_features.append(self.norm(x))

        x = self.norm(x)

        # Global pooling
        pooled = x.mean(dim=1) # (B, embed_dim)
        return pooled, intermediate_features, x