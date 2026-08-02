from sympy import false
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer"""
    def __init__(
            self, 
            in_features:int, 
            out_features:int, 
            num_heads:int=1,  
            dropout:float=0.0, 
            concat:bool=True, 
            negative_slope:float=0.2
    ):
        super().__init_()
        assert in_features > 0
        assert out_features > 0
        assert num_heads > 0

        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.negative_slope = negative_slope



        # self.W = nn.Parameter(torch.zeros(num_heads, in_features))
        # self.a = nn.Parameter(torch.zeros(num_heads, 2*out_features, 1))

        self.lin = nn.Linear(in_features, num_heads*out_features, bias=False)
        self.attn_src = nn.Parameter(torch.empty(num_heads, out_features))
        self.attn_dst = nn.Parameter(torch.empty(num_heads, out_features))

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.atn_src.unsqueeze(-1))
        nn.init.xavier_uniform_(self.attn_dst.unsqueeze(-1))

    def forward(self, h:torch.Tensor, adj:Optional[torch.Tensor]=None)->torch.Tensor:
        # h: (B, N, in_features)
        # adj: (B, N, N) or None (fully connected graph or no adjacency information)
        # B, N, _ = h.size()
        squeeze_batch = False
        if h.dim() == 2:
            h = h.unsqueeze(0)  # (1, N, in_features)
            squeeze_batch = True

        B, N, _ = h.shape


        x = self.lin(h).view(B, N, self.num_heads, self.out_features)  # (B, H, N, F)


        for head in range(self.num_heads):
            Wh = torch.mm(h, self.W[head].unsqueeze(1)) # (N, out_features)
            a_input = torch.cat([Wh.repeat(1, N).view(N*N, -1), Wh.repeat(N, 1)], dim=1).view(N, -1, 2*self.out_features) # (N, N, 2*out_features)
            e = self.leaky_relu(torch.matmul(a_input, self.a[head]).squeeze(2)) # (N, N)

            if adj is not None:
                if adj.dim() == 2:
                    adj = adj.unsqueeze(0)
                adj = adj.to(dtype=torch.bool)

                if adj.shape[0] == 1 and B > 1:
                    adj = adj.expand(B, -1, -1)

                e = e.masked_fill(~adj[:, None, :, :], float("-inf"))


            attn = F.softmax(e, dim=-1) # (N, N)
            attn = self.dropout(attn)

            h_prime = torch.matmul(attn, x) # (B, H, N, F)

        if self.concat:
            # return F.elu(h_prime) # (N, num_heads*out_features)
            h_prime = h_prime.transpose(1,2).contiguous().view(B, N, self.num_heads*self.out_features) # (B, N, num_heads*out_features)
        else:
            h_prime =  h_prime.mean(dim=1) # (B, N, out_features)

        if squeeze_batch:
            out = h_prime.squeeze(0)  # (N, num_heads*out_features) or (N, out_features)
        return out 

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
    def __init__(self, in_features:int, hidden_dim: Optional[int]=None, dropout:float=0.0):
        super().__init__()
        hidden_dim = hidden_dim or max(1, in_features// 2)
        self.gate = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),            
        )

    def forward(self, h:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
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

        self.initial_proj = nn.Linear(config.node_features, config.gat_hidden_features)

        layers = []
        in_dim = config.gat_hidden_dim

        for i in range(config.gat_num_layers):
            last =  i == config.gat_num_layers - 1
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
        self.gated_pool = GatedGraphPooling(in_dim, hidden_dim = getattr(config, "pool_hidden_dim", None), dropout = config.gat_dropout)

    def forward(
            self, 
            x:torch.Tensor, 
            adj:Optional[torch.Tensor]=None,
            return_intermediate:bool=True
        )->Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        # x: (B, N, node_features)
        # adj: (B, N, N) or None
        h = self.initial_proj(x)  # (B, N, gat_hidden_dim)

        intermediate_features: List[torch.Tensor] = []

        for i, layer in enumerate(self.layers, start=1):
            h = layer(h, adj)  # (B, N, out_features)
            # x = layer(x, adj)  # (B, N, out_features)
            # if i+1 in self.config.functional_stages:
            #     intermediate_features.append(h)
            if return_intermediate and hasattr(self.config, 'functional_stages') and i in self.config.functional_stages:
                intermediate_features.append(h)
            

        # gated pooling and fusion projection
        pooled_h, _ = self.gated_pool(h)  # (B, fusion_dim)
        # project to fusion dimension
        fused_h = self.fusion_proj(pooled_h)  # (B, fusion_dim)
        
        return fused_h, intermediate_features, h