from dataclasses import dataclass
from typing import Optional

import pytest
import torch 

from NeuroFusion.models.gat import (
    GraphAttentionNetwork,
    GraphAttentionLayer,
    GatedGraphPooling
)

from NeuroFusion.models.vit3d import (
    PatchEmbed3D,
    Attention3D,
    Mlp3D,
    Block3D,
    VisionTransformer3D
)


@dataclass
class Config:
    # vit parameters
    img_size: int = 32
    patch_size: int = 4
    in_channels: int = 1
    embed_dim: int = 64
    num_heads: int = 4
    mlp_ratio: float = 4.0 
    num_layers: int = 6
    dropout: float = 0.1
    cross_attention: bool = True
    cross_attention_heads: int = 4
    structural_stages: list[int] = None

    # gat parameters
    node_features: int = 8
    gat_hidden_dim: int = 16
    gat_out_dim: int = 12
    gat_num_layers: int = 3
    gat_num_heads: int = 4
    gat_dropout: float = 0.1
    fusion_dim: int = 10
    functional_stages: list[int] = None

    def __post_init__(self):
        if self.functional_stages is None:
            self.functional_stages = list(range(1, self.gat_num_layers + 1))
        if self.structural_stages is None:          
            self.structural_stages = [2,4,6]  # Example stages for structural features  

# ==== utility fixtures for testing ====
@pytest.fixture
def device():
    """Get appropriate device for testing"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def seed_fixture():
    def _set_seed(seed=42):
        torch.manual_seed(seed)
        import numpy as np
        np.random.seed(seed)
    return _set_seed


# ======= model configuration fixture
@pytest.fixture
def model_config():
    return Config()


@pytest.fixture
def get_patchEmbed3D(model_config):
    return PatchEmbed3D(
        img_size=model_config.img_size,
        patch_size=model_config.patch_size,
        in_chans=model_config.in_channels,
        embed_dim=model_config.embed_dim
    )

@pytest.fixture
def get_attention3D(model_config):
    return Attention3D(
        dim=model_config.embed_dim,
        num_heads=model_config.num_heads
    )


@pytest.fixture
def get_mlp3D(model_config):
    return Mlp3D(
        in_features=model_config.embed_dim,
        hidden_features=int(model_config.embed_dim * model_config.mlp_ratio)
    )

@pytest.fixture
def get_block3D(model_config):
    return Block3D(
        dim=model_config.embed_dim,
        num_heads=model_config.num_heads,
        mlp_ratio=model_config.mlp_ratio
    )

@pytest.fixture
def get_vit_model(model_config):
    return VisionTransformer3D(model_config)




@pytest.fixture
def gat_model(model_config):
    return GraphAttentionNetwork(model_config)


@pytest.fixture
def sample_node_features(model_config):
    batch_size = 2
    num_nodes = 10
    return torch.randn(batch_size, num_nodes, model_config.node_features)
    

@pytest.fixture
def sample_adj_matrix():
    batch_size = 2
    num_nodes = 10
    adj = torch.randint(0, 2, (batch_size, num_nodes, num_nodes)).float()
    return adj

