from sympy import false
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer"""
    def __init__(self, in_features:int, out_features:int, num_heads:int=1,  dropout:float=0.0, concat:bool=True, negative_slope:float=0.2):
        super().__init_()
        self.num_heads = num_heads
        self.out_features = out_features
        self.concat = concat
        self.negative_slope = negative_slope



        self.W = nn.Parameter(torch.zeros(num_heads, in_features))
        self.a = nn.Parameter(torch.zeros(num_heads, 2*out_features, 1))

        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h:torch.Tensor, adj:Optional[torch.Tensor]=None)->torch.Tensor:
        # h: (B, N, in_features)
        # adj: (B, N, N) or None (fully connected graph or no adjacency information)
        # B, N, _ = h.size()
        N = h.size(0)


        h_prime = torch.zeros(N, self.num_heads*self.out_features).to(h.device)

        for head in range(self.num_heads):
            Wh = torch.mm(h, self.W[head].unsqueeze(1)) # (N, out_features)
            a_input = torch.cat([Wh.repeat(1, N).view(N*N, -1), Wh.repeat(N, 1)], dim=1).view(N, -1, 2*self.out_features) # (N, N, 2*out_features)
            e = self.leaky_relu(torch.matmul(a_input, self.a[head]).squeeze(2)) # (N, N)

            if adj is not None:
                e = e.masked_fill(adj == 0, float('-inf'))

            attention = F.softmax(e, dim=1) # (N, N)
            attention = self.dropout(attention)

            h_prime[:, head*self.out_features:(head+1)*self.out_features] = torch.matmul(attention, Wh) # (N, out_features)

        if self.concat:
            # return F.elu(h_prime) # (N, num_heads*out_features)
            h_prime = F.elu(h_prime) # (B, N, num_heads*out_features)
        else:
            h_prime =  h_prime.mean(dim=1) # (B, N, out_features)

        return h_prime

    # def forward(self, h: torch.Tensor, adj: Optional[torch.Tensor] = None) -> torch.Tensor:
    #     # h: (B, N, in_features)
    #     # adj: (B, N, N) or None (fully connected)
    #     B, N, _ = h.shape
        
    #     # Compute attention
    #     Wh = torch.einsum('bni,hio->bhno', h, self.W)  # (B, H, N, out_features)
        
    #     # Compute attention coefficients
    #     a_input = torch.cat([
    #         Wh.unsqueeze(3).expand(-1, -1, -1, N, -1),
    #         Wh.unsqueeze(2).expand(-1, -1, N, -1, -1)
    #     ], dim=-1)  # (B, H, N, N, 2*out_features)
        
    #     e = torch.matmul(a_input, self.a).squeeze(-1)  # (B, H, N, N)
    #     e = self.leaky_relu(e)
        
    #     # Mask attention (if adjacency matrix provided)
    #     if adj is not None:
    #         # adj: (B, N, N) with 0/1 or weights
    #         mask = (adj > 0).unsqueeze(1).float()  # (B, 1, N, N)
    #         e = e.masked_fill(mask == 0, float('-inf'))
        
    #     attention = F.softmax(e, dim=-1)  # (B, H, N, N)
    #     attention = self.dropout(attention)
        
    #     # Apply attention
    #     h_prime = torch.matmul(attention, Wh)  # (B, H, N, out_features)
        
    #     if self.concat:
    #         # Concatenate heads
    #         h_prime = h_prime.transpose(1, 2).contiguous().view(B, N, -1)  # (B, N, H*out_features)
    #     else:
    #         # Average heads
    #         h_prime = h_prime.mean(dim=1)  # (B, N, out_features)
        
    #     return h_prime

class GatedGraphPooling(nn.Module):
    """Gated Graph Pooling Layer"""
    def __init__(self, in_features:int, hidden_dim: Optional[int]=None):
        super().__init__()
        hidden_dim = hidden_dim or in_features// 2
        self.gate = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, h:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        # h: (B, N, in_features)
        gate_values = torch.sigmoid(self.gate(h))  # (B, N, 1)
        pooled_h = torch.sum(gate_values * h, dim=1)  # (B, in_features)
        return pooled_h, gate_values

class GraphAttentionNetwork(nn.Module):
    """Multi-layer Graph Attention Network with hierarchical features"""
    def __init__(self, config:"ModelConfig"):
        super().__init__()
        self.config = config

        self.initial_proj = nn.Linear(config.node_features, config.gat_hidden_features)

        self.layers = nn.ModuleList([])
        in_dim = config.gat_hidden_dim

        for i in range(config.gat_num_layers):
            if i == config.gat_num_layers - 1:
                out_dim = config.gat_out_dim
                num_heads = 1  # Last layer has a single head
                concat = False  # No concatenation in the last layer
            else:
                out_dim = config.gat_hidden_dim // config.gat_num_heads
                num_heads = config.gat_num_heads
                concat = True  # Concatenate in intermediate layers

            self.layers.append(
                GraphAttentionLayer(
                    in_features=in_dim,
                    out_features=out_dim,
                    num_heads=num_heads,
                    dropout=config.gat_dropout,
                    concat=concat
                )
            )
            if i < config.gat_num_layers - 1 and concat:
                in_dim = out_dim * num_heads
            else:
                in_dim = out_dim

        # self.layers = nn.ModuleList([
        #     GraphAttentionLayer(
        #         in_features=in_dim,
        #         out_features=config.hidden_features,
        #         num_heads=config.num_heads,
        #         dropout=config.dropout,
        #         concat=True
        #     ) for i in range(config.num_layers)
        # ])


        self.fusion_proj = nn.Linear(config.gat_hidden_dim, config.fusion_dim)
        self.gated_pool = GatedGraphPooling(config.gat_hidden_dim)

    def forward(self, x:torch.Tensor, adj:Optional[torch.Tensor]=None)->Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        # x: (B, N, node_features)
        # adj: (B, N, N) or None
        h = self.initial_proj(x)  # (B, N, gat_hidden_dim)

        intermediate_features = []

        for i, layer in enumerate(self.layers):
            h = layer(h, adj)  # (B, N, out_features)
            # x = layer(x, adj)  # (B, N, out_features)
            if i+1 in self.config.functional_stages:
                intermediate_features.append(h)
            

        # gated pooling and fusion projection
        pooled_h, _ = self.gated_pool(h)  # (B, fusion_dim)
        # project to fusion dimension
        fused_h = self.fusion_proj(pooled_h)  # (B, fusion_dim)
        
        return fused_h, intermediate_features, h