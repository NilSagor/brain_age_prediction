# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import List, Optional, Tuple


# class GraphAttentionLayer(nn.Module):
#     """Graph Attention Layer (multi-head)"""
#     def __init__(
#         self,
#         in_features: int,
#         out_features: int,
#         num_heads: int = 1,
#         dropout: float = 0.0,
#         concat: bool = True,
#         negative_slope: float = 0.2,
#     ):
#         super().__init__()
#         assert in_features > 0
#         assert out_features > 0
#         assert num_heads > 0

#         self.in_features = in_features
#         self.out_features = out_features
#         self.num_heads = num_heads
#         self.concat = concat
#         self.negative_slope = negative_slope

#         # Weight matrix per head: (num_heads, in_features, out_features)
#         self.W = nn.Parameter(torch.empty(num_heads, in_features, out_features))
#         # Attention parameters per head: (num_heads, 2 * out_features, 1)
#         self.a = nn.Parameter(torch.empty(num_heads, 2 * out_features, 1))

#         self.dropout = nn.Dropout(dropout)
#         self.leaky_relu = nn.LeakyReLU(negative_slope)
#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.W)
#         nn.init.xavier_uniform_(self.a)

#     def forward(self, h: torch.Tensor, adj: Optional[torch.Tensor] = None) -> torch.Tensor:
#         # h: (B, N, in_features)
#         # adj: (B, N, N) or None (fully connected)
#         B, N, _ = h.shape

#         # Compute linear transformation per head: (B, H, N, out_features)
#         Wh = torch.einsum('bni,hio->bhno', h, self.W)  # (B, H, N, out_features)

#         # Compute attention coefficients
#         # Concatenate source and target features per head: (B, H, N, N, 2*out_features)
#         a_input = torch.cat([
#             Wh.unsqueeze(3).expand(-1, -1, -1, N, -1),   # (B, H, N, N, F)
#             Wh.unsqueeze(2).expand(-1, -1, N, -1, -1),   # (B, H, N, N, F)
#         ], dim=-1)  # (B, H, N, N, 2*F)

#         e = torch.matmul(a_input, self.a).squeeze(-1)  # (B, H, N, N)
#         e = self.leaky_relu(e)

#         # Mask with adjacency matrix (if provided)
#         if adj is not None:
#             # adj: (B, N, N) with 1 for edges, 0 otherwise
#             if adj.dim() == 2:
#                 adj = adj.unsqueeze(0)
#             # Ensure batch dimension matches
#             if adj.shape[0] == 1 and B > 1:
#                 adj = adj.expand(B, -1, -1)
#             mask = (adj > 0).unsqueeze(1).float()  # (B, 1, N, N)
#             e = e.masked_fill(mask == 0, float('-inf'))

#         attention = F.softmax(e, dim=-1)  # (B, H, N, N)
#         attention = self.dropout(attention)

#         # Apply attention: (B, H, N, out_features)
#         h_prime = torch.matmul(attention, Wh)

#         if self.concat:
#             # Concatenate heads: (B, N, H*out_features)
#             h_prime = h_prime.transpose(1, 2).contiguous().view(B, N, -1)
#         else:
#             # Average heads: (B, N, out_features)
#             h_prime = h_prime.mean(dim=1)

#         return h_prime


# class GatedGraphPooling(nn.Module):
#     """Gated graph pooling with learnable importance scores"""
#     def __init__(self, in_features: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
#         super().__init__()
#         hidden_dim = hidden_dim or max(1, in_features // 2)
#         self.gate = nn.Sequential(
#             nn.Linear(in_features, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, 1),
#         )

#     def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         # h: (B, N, in_features)
#         gate_logits = self.gate(h)          # (B, N, 1)
#         gate_values = torch.sigmoid(gate_logits)  # (B, N, 1)
#         pooled = torch.sum(gate_values * h, dim=1)  # (B, in_features)
#         return pooled, gate_values


# class GraphAttentionNetwork(nn.Module):
#     """Multi-layer Graph Attention Network with hierarchical features"""
#     def __init__(self, config):
#         super().__init__()
#         self.config = config

#         # Initial projection to hidden dimension
#         self.initial_proj = nn.Linear(config.node_features, config.gat_hidden_dim)

#         layers = []
#         in_dim = config.gat_hidden_dim

#         for i in range(config.gat_num_layers):
#             last = (i == config.gat_num_layers - 1)
#             if last:
#                 out_dim = config.gat_out_dim
#                 heads = 1
#                 concat = False
#             else:
#                 heads = config.gat_num_heads
#                 out_dim = config.gat_hidden_dim // heads
#                 concat = True

#             layers.append(
#                 GraphAttentionLayer(
#                     in_features=in_dim,
#                     out_features=out_dim,
#                     num_heads=heads,
#                     dropout=config.gat_dropout,
#                     concat=concat,
#                 )
#             )
#             # Update input dimension for next layer
#             in_dim = out_dim * heads if concat else out_dim

#         self.layers = nn.ModuleList(layers)

#         # Project to fusion dimension (for cross-modal fusion)
#         self.fusion_proj = nn.Linear(in_dim, config.fusion_dim)

#         # Gated pooling
#         self.gated_pool = GatedGraphPooling(
#             in_dim,
#             hidden_dim=getattr(config, 'pool_hidden_dim', None),
#             dropout=config.gat_dropout
#         )

#     def forward(
#         self,
#         x: torch.Tensor,
#         adj: Optional[torch.Tensor] = None,
#         return_intermediate: bool = True
#     ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
#         # x: (B, N, node_features)
#         # adj: (B, N, N) or None
#         h = self.initial_proj(x)  # (B, N, gat_hidden_dim)

#         intermediate_features: List[torch.Tensor] = []

#         for i, layer in enumerate(self.layers, start=1):
#             h = layer(h, adj)  # (B, N, out_dim)
#             if return_intermediate and hasattr(self.config, 'functional_stages') and i in self.config.functional_stages:
#                 intermediate_features.append(h)

#         # Gated pooling and fusion projection
#         pooled_h, _ = self.gated_pool(h)                # (B, in_dim)
#         fused_h = self.fusion_proj(pooled_h)            # (B, fusion_dim)

#         return fused_h, intermediate_features, h




# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import List, Optional, Tuple


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer"""
    def __init__(
            self, 
            in_features: int, 
            out_features: int, 
            num_heads: int = 1,  
            dropout: float = 0.0, 
            concat: bool = True, 
            negative_slope: float = 0.2
    ):
        super().__init__()
        assert in_features > 0
        assert out_features > 0
        assert num_heads > 0

        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.negative_slope = negative_slope

        self.lin = nn.Linear(in_features, num_heads * out_features, bias=False)
        self.attn_src = nn.Parameter(torch.empty(num_heads, out_features))
        self.attn_dst = nn.Parameter(torch.empty(num_heads, out_features))

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, h: torch.Tensor, adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        # h: (B, N, in_features) or (N, in_features)
        # adj: (B, N, N) or (N, N) or None
        squeeze_batch = False
        if h.dim() == 2:
            h = h.unsqueeze(0)  # (1, N, in_features)
            squeeze_batch = True

        B, N, _ = h.shape

        # Linear projection and reshape to (B, H, N, F)
        Wh = self.lin(h).view(B, N, self.num_heads, self.out_features).permute(0, 2, 1, 3)

        # Additive attention: e_{ij} = LeakyReLU(Wh_i @ a_src + Wh_j @ a_dst)
        src_logits = torch.einsum('bhnf,hf->bhn', Wh, self.attn_src)  # (B, H, N)
        dst_logits = torch.einsum('bhnf,hf->bhn', Wh, self.attn_dst)  # (B, H, N)
        e = src_logits.unsqueeze(-1) + dst_logits.unsqueeze(-2)       # (B, H, N, N)
        e = self.leaky_relu(e)

        # Mask attention (if adjacency matrix provided)
        if adj is not None:
            if adj.dim() == 2:
                adj = adj.unsqueeze(0)  # (1, N, N)
            adj = adj.to(dtype=torch.bool)

            # Add self-loops so no row is ever fully masked (prevents NaN in softmax)
            eye = torch.eye(N, device=adj.device, dtype=torch.bool).unsqueeze(0)
            adj = adj | eye

            if adj.shape[0] == 1 and B > 1:
                adj = adj.expand(B, -1, -1)

            e = e.masked_fill(~adj.unsqueeze(1), float('-inf'))

        # Softmax and dropout
        attn = F.softmax(e, dim=-1)  # (B, H, N, N)
        attn = self.dropout(attn)
        self.attn_weights = attn     # stored for inspection/tests

        # Apply attention
        h_prime = torch.einsum('bhnm,bhmf->bhnf', attn, Wh)  # (B, H, N, F)

        if self.concat:
            # Concatenate heads: (B, H, N, F) -> (B, N, H*F)
            h_prime = h_prime.permute(0, 2, 1, 3).contiguous().view(B, N, self.num_heads * self.out_features)
        else:
            # Average heads: (B, H, N, F) -> (B, N, F)
            h_prime = h_prime.mean(dim=1)

        if squeeze_batch:
            h_prime = h_prime.squeeze(0)

        return h_prime


class GatedGraphPooling(nn.Module):
    """Gated Graph Pooling Layer"""
    def __init__(self, in_features: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or max(1, in_features // 2)
        self.gate = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),            
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h: (B, N, in_features)
        gate_logits = self.gate(h)  # (B, N, 1)
        gate_values = torch.sigmoid(gate_logits)  # (B, N, 1)
        pooled_h = torch.sum(gate_values * h, dim=1)  # (B, in_features)
        return pooled_h, gate_values


class GraphAttentionNetwork(nn.Module):
    """Multi-layer Graph Attention Network with hierarchical features"""
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Fixed: config has gat_hidden_dim, not gat_hidden_features
        self.initial_proj = nn.Linear(config.node_features, config.gat_hidden_dim)

        layers = []
        in_dim = config.gat_hidden_dim

        for i in range(config.gat_num_layers):
            last = i == config.gat_num_layers - 1
            if last:
                out_dim = config.gat_out_dim
                heads = 1  # Last layer has a single head
                concat = False  # No concatenation in the last layer
            else:
                heads = config.gat_num_heads
                out_dim = config.gat_hidden_dim // heads
                concat = True  # Concatenate in intermediate layers

            layers.append(
                GraphAttentionLayer(
                    in_features=in_dim,
                    out_features=out_dim,
                    num_heads=heads,
                    dropout=config.gat_dropout,
                    concat=concat,
                )
            )

            in_dim = out_dim * heads if concat else out_dim

        self.layers = nn.ModuleList(layers)
        self.fusion_proj = nn.Linear(in_dim, config.fusion_dim)
        self.gated_pool = GatedGraphPooling(
            in_dim, 
            hidden_dim=getattr(config, "pool_hidden_dim", None), 
            dropout=config.gat_dropout
        )

    def forward(
            self, 
            x: torch.Tensor, 
            adj: Optional[torch.Tensor] = None,
            return_intermediate: bool = True
        ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        # x: (B, N, node_features)
        # adj: (B, N, N) or None
        h = self.initial_proj(x)  # (B, N, gat_hidden_dim)

        intermediate_features: List[torch.Tensor] = []

        for i, layer in enumerate(self.layers, start=1):
            h = layer(h, adj)  # (B, N, out_features)
            if return_intermediate and hasattr(self.config, 'functional_stages') and i in self.config.functional_stages:
                intermediate_features.append(h)

        # gated pooling and fusion projection
        pooled_h, _ = self.gated_pool(h)  # (B, in_dim)
        fused_h = self.fusion_proj(pooled_h)  # (B, fusion_dim)
        
        # Return order matches tests: (node_output, intermediates, fused/pooled_output)
        return h, intermediate_features, fused_h