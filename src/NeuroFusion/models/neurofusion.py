# src/NeuroFusion/models/neurofusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple

from .vit3d import VisionTransformer3D
from .gat import GraphAttentionNetwork
from .fusion import HierarchicalFusionBlock

from NeuroFusion.config import ModelConfig, TrainingConfig


class NeuroFusion(nn.Module):
    """Complete NeuroFusion model with hierarchical fusion and supervision"""
    def __init__(self, config: 'ModelConfig'):
    
        super().__init__()
        self.config = config
        
        # Modality-specific encoders
        self.structural_encoder = VisionTransformer3D(config)
        self.functional_encoder = GraphAttentionNetwork(config)
        
        # Projection layers for intermediate features to common dimension
        # self.structural_proj = nn.ModuleList([
        #     nn.Linear(config.embed_dim, config.fusion_dim)
        #     for _ in range(len(config.structural_stages))
        # ])
        # self.functional_proj = nn.ModuleList([
        #     nn.Linear(config.gat_hidden_dim, config.fusion_dim)
        #     for _ in range(len(config.functional_stages))
        # ])

        # Projection layers for intermediate features to common dimension
        self.structural_proj = nn.ModuleList([
            nn.Linear(config.embed_dim, config.fusion_dim)
            for _ in range(len(config.structural_stages))
        ])
        
        # Functional intermediates: non-last layers have gat_hidden_dim, last layer has gat_out_dim
        functional_in_dims = []
        for stage in config.functional_stages:
            if stage == config.gat_num_layers:
                functional_in_dims.append(config.gat_out_dim)
            else:
                functional_in_dims.append(config.gat_hidden_dim)
        
        self.functional_proj = nn.ModuleList([
            nn.Linear(in_dim, config.fusion_dim)
            for in_dim in functional_in_dims
        ])
        
        # Hierarchical fusion blocks
        self.fusion_blocks = nn.ModuleList([
            HierarchicalFusionBlock(config.fusion_dim, config.cross_attn_heads)
            for _ in range(len(config.structural_stages))
        ])
        
        # Auxiliary regression heads for hierarchical supervision
        self.aux_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.fusion_dim, config.fusion_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(config.fusion_dim // 2, 1)
            )
            for _ in range(len(config.structural_stages))
        ])
        
        # Final regression head
        self.final_head = nn.Sequential(
            nn.Linear(config.fusion_dim, config.fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.fusion_dim // 2, 1)
        )
        
        # Store intermediate features (optional)
        self.structural_features = []
        self.functional_features = []
        self.fused_features = []

    def forward(self, sMRI: torch.Tensor, fMRI: torch.Tensor, 
                adj: Optional[torch.Tensor] = None,
                return_all: bool = True) -> Dict[str, torch.Tensor]:
        """
        Args:
            sMRI: (B, 1, D, H, W) structural MRI volume
            fMRI: (B, N_nodes, N_features) functional connectivity features
            adj: (B, N_nodes, N_nodes) adjacency matrix (optional)
            return_all: Return all hierarchical outputs
        
        Returns:
            Dictionary with predictions and intermediate features
        """
        # 1. Structural representation learning
        structural_pooled, structural_intermediate, structural_tokens = self.structural_encoder(sMRI)
        
        # 2. Functional representation learning
        functional_pooled, functional_intermediate, functional_tokens = self.functional_encoder(fMRI, adj)
        
        # 3. Hierarchical fusion
        fused_features = []
        aux_predictions = []
        gates = []
        
        for i in range(len(self.config.structural_stages)):
            # Project features to common dimension
            s_feat = self.structural_proj[i](structural_intermediate[i].mean(dim=1))
            f_feat = self.functional_proj[i](functional_intermediate[i].mean(dim=1))
            
            # Fusion modules expect 3D inputs (B, N, D) — add sequence dim
            if s_feat.dim() == 2:
                s_feat = s_feat.unsqueeze(1)
            if f_feat.dim() == 2:
                f_feat = f_feat.unsqueeze(1)


            # Fusion
            fused, gate = self.fusion_blocks[i](s_feat, f_feat)

            # Squeeze back to 2D for regression heads
            fused = fused.squeeze(1)
            gate = gate.squeeze(1)

            fused_features.append(fused)
            gates.append(gate)
            
            # Auxiliary prediction
            aux_pred = self.aux_heads[i](fused)
            aux_predictions.append(aux_pred.squeeze(-1))
        
        # 4. Final prediction using deepest fused representation
        final_fused = fused_features[-1]
        final_combined = final_fused + functional_tokens
        final_pred = self.final_head(final_combined).squeeze(-1)
        
        # Store for later use
        self.structural_features = structural_intermediate
        self.functional_features = functional_intermediate
        self.fused_features = fused_features
        
        # Return outputs
        output = {
            'final_prediction': final_pred,
            'aux_predictions': aux_predictions,  # List of predictions for each stage
            'fused_features': fused_features,
            'gates': gates,
        }
        
        if not return_all:
            # During inference, only return final prediction
            return {'final_prediction': final_pred}
        
        return output

    def compute_loss(self, batch: Dict[str, torch.Tensor], 
                     config: 'TrainingConfig') -> Dict[str, torch.Tensor]:
        """Compute total loss with hierarchical supervision"""
        sMRI = batch['sMRI']
        fMRI = batch['fMRI']
        age = batch['age']
        adj = batch.get('adj', None)
        
        # Forward pass
        outputs = self.forward(sMRI, fMRI, adj, return_all=True)
        
        # Huber loss for main prediction
        main_loss = F.huber_loss(outputs['final_prediction'], age)
        
        # Auxiliary losses
        aux_losses = []
        for i, aux_pred in enumerate(outputs['aux_predictions']):
            if i < len(self.config.aux_weights):
                aux_loss = F.huber_loss(aux_pred, age)
                aux_losses.append(self.config.aux_weights[i] * aux_loss)
        
        total_loss = main_loss + sum(aux_losses)
        
        return {
            'loss': total_loss,
            'main_loss': main_loss,
            'aux_losses': aux_losses,
            'final_prediction': outputs['final_prediction'],
            'aux_predictions': outputs['aux_predictions'],
        }

    def inference(self, sMRI: torch.Tensor, fMRI: torch.Tensor, 
                  adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Inference mode: only forward pass with final prediction"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(sMRI, fMRI, adj, return_all=False)
        return outputs['final_prediction']    