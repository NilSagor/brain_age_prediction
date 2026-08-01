import pytest
import torch 
from NeuroFusion.models.gat import GraphAttentionNetwork, GraphAttentionLayer, GatedGraphPooling

class TestGraphAttentionNetwork:
    """Tests for Graph Attention Network"""
    def test_forward_shape(self, model_config):
        gat = GraphAttentionNetwork(model_config)

        batch_size = 2
        num_nodes = 10
        x = torch.randn(batch_size, num_nodes, model_config.node_features)
        adj = torch.randint(0, 2, (batch_size, num_nodes, num_nodes)).float()

        output, intermediate_features, pooled_output = gat(x, adj)
        assert output.shape == (batch_size, num_nodes, model_config.gat_out_dim)
        assert len(intermediate_features) == model_config.gat_num_layers
        assert pooled_output.shape == (batch_size, model_config.fusion_dim)

    def test_forward_shape_no_adj(self, model_config):
        gat = GraphAttentionNetwork(model_config)

        batch_size = 2
        num_nodes = 10
        x = torch.randn(batch_size, num_nodes, model_config.node_features)

        output, intermediate_features, pooled_output = gat(x)
        assert output.shape == (batch_size, num_nodes, model_config.gat_out_dim)
        assert len(intermediate_features) == model_config.gat_num_layers
        assert pooled_output.shape == (batch_size, model_config.fusion_dim)

    def test_gradient_flow(self, model_config):
        gat = GraphAttentionNetwork(model_config)

        batch_size = 2
        num_nodes = 10
        x = torch.randn(batch_size, num_nodes, model_config.node_features, requires_grad=True)
        adj = torch.randint(0, 2, (batch_size, num_nodes, num_nodes)).float()

        output, intermediate_features, pooled_output = gat(x, adj)
        loss = output.sum() + pooled_output.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all(), "Gradient contains NaN or Inf values"


    def test_intermediate_features(self, model_config):
        gat = GraphAttentionNetwork(model_config)

        batch_size = 2
        num_nodes = 10
        x = torch.randn(batch_size, num_nodes, model_config.node_features)
        adj = torch.randint(0, 2, (batch_size, num_nodes, num_nodes)).float()

        output, intermediate_features, pooled_output = gat(x, adj)
        for i, features in enumerate(intermediate_features):
            expected_dim = model_config.gat_hidden_dim if i < model_config.gat_num_layers - 1 else model_config.gat_out_dim
            assert features.shape == (batch_size, num_nodes, expected_dim)

    def test_attention_weights(self, model_config):
        gat = GraphAttentionNetwork(model_config)

        batch_size = 2
        num_nodes = 10
        x = torch.randn(batch_size, num_nodes, model_config.node_features)
        adj = torch.randint(0, 2, (batch_size, num_nodes, num_nodes)).float()

        output, intermediate_features, pooled_output = gat(x, adj)
        # Check if attention weights are in the expected range [0, 1]
        for layer in gat.layers:
            attn_weights = layer.attn_weights
            assert torch.all(attn_weights >= 0) and torch.all(attn_weights <= 1)
    def test_parameter_count(self, gat_model):
        num_params = sum(p.numel() for p in gat_model.parameters())
        assert num_params > 0, "Model should have parameters"
        assert num_params < 1e7, "Model should not have an excessive number of parameters"