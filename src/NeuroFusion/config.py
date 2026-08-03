# config.py
from dataclasses import dataclass
from typing import Tuple, List, Optional

@dataclass
class ModelConfig:
    # Structural branch (3D ViT)
    img_size: int = 96
    patch_size: int = 16
    in_channels: int = 1
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    
    # Functional branch (GAT)
    num_nodes: int = 90
    node_features: int = 90
    gat_hidden_dim: int = 256
    gat_out_dim: int = 256        # ← ADD THIS
    gat_num_layers: int = 6
    gat_num_heads: int = 8
    gat_dropout: float = 0.1
    
    # Fusion
    fusion_dim: int = 256
    cross_attn_heads: int = 8
    
    # Hierarchical stages
    structural_stages: List[int] = (4, 8, 12)
    functional_stages: List[int] = (2, 4, 6)
    
    # Auxiliary weights
    aux_weights: List[float] = (0.3, 0.5, 0.7)

@dataclass
class TrainingConfig:
    batch_size: int = 8
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    seed: int = 42
    
    # Data
    data_dir: str = "./data"
    num_workers: int = 4
    val_split: float = 0.1
    num_folds: int = 5