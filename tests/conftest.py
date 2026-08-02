from dataclasses import dataclass
from typing import Optional

import pytest
import torch 

from NeuroFusion.models.gat import (
    GraphAttentionNetwork,
    GraphAttentionLayer,
    GatedGraphPooling
)

@dataclass
class Config:
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



@pytest.fixture
def seed_fixture():
    def _set_seed(seed=42):
        torch.manual_seed(seed)
        import numpy as np
        np.random.seed(seed)
    return _set_seed

@pytest.fixture
def model_config():
    return Config()

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