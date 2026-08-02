import pytest
import torch

from NeuroFusion.models.gat import GraphAttentionNetwork, GraphAttentionLayer, GatedGraphPooling


# class TestGraphAttentionNetwork:
#     """Tests for Graph Attention Network"""

#     def test_forward_shape(self, model_config, sample_node_features, sample_adj_matrix):
#         gat = GraphAttentionNetwork(model_config)
#         fused_output, intermediate_features, node_features = gat(
#             sample_node_features, sample_adj_matrix
#         )

#         batch_size, num_nodes, _ = sample_node_features.shape

#         # fused_output is pooled and projected to fusion dimension
#         assert fused_output.shape == (batch_size, model_config.fusion_dim)

#         # node_features are final node embeddings: (B, N, gat_out_dim)
#         assert node_features.shape == (batch_size, num_nodes, model_config.gat_out_dim)

#         # intermediate features should match functional_stages length
#         assert len(intermediate_features) == len(model_config.functional_stages)

#         # Check each intermediate feature shape
#         for i, stage in enumerate(model_config.functional_stages):
#             expected_dim = model_config.gat_hidden_dim if stage < model_config.gat_num_layers else model_config.gat_out_dim
#             # For intermediate layers before last, output dim = gat_hidden_dim
#             # For last layer, output dim = gat_out_dim
#             # But functional_stages can be any subset; we can check shape roughly.
#             # Let's just check batch and nodes.
#             assert intermediate_features[i].shape[0] == batch_size
#             assert intermediate_features[i].shape[1] == num_nodes

#     def test_forward_shape_no_adj(self, model_config):
#         gat = GraphAttentionNetwork(model_config)

#         batch_size = 2
#         num_nodes = 10
#         x = torch.randn(batch_size, num_nodes, model_config.node_features)

#         fused_output, intermediate_features, node_features = gat(x, adj=None)

#         assert fused_output.shape == (batch_size, model_config.fusion_dim)
#         assert node_features.shape == (batch_size, num_nodes, model_config.gat_out_dim)
#         assert len(intermediate_features) == len(model_config.functional_stages)

#     def test_gradient_flow(self, model_config):
#         gat = GraphAttentionNetwork(model_config)

#         batch_size = 2
#         num_nodes = 10
#         x = torch.randn(batch_size, num_nodes, model_config.node_features, requires_grad=True)
#         adj = torch.randint(0, 2, (batch_size, num_nodes, num_nodes)).float()

#         fused_output, _, node_features = gat(x, adj)
#         loss = fused_output.sum() + node_features.sum()
#         loss.backward()

#         assert x.grad is not None
#         assert torch.isfinite(x.grad).all(), "Gradient contains NaN or Inf values"

#         # Check that all trainable parameters have gradients
#         for param in gat.parameters():
#             if param.requires_grad:
#                 assert param.grad is not None
#                 assert torch.isfinite(param.grad).all()

#     def test_intermediate_features_hierarchy(self, model_config):
#         gat = GraphAttentionNetwork(model_config)

#         batch_size = 2
#         num_nodes = 10
#         x = torch.randn(batch_size, num_nodes, model_config.node_features)
#         adj = torch.randint(0, 2, (batch_size, num_nodes, num_nodes)).float()

#         _, intermediate_features, _ = gat(x, adj)

#         # Check that features evolve across layers (different values)
#         for i in range(len(intermediate_features) - 1):
#             # They should not be identical
#             assert not torch.allclose(intermediate_features[i], intermediate_features[i+1], rtol=1e-3)

#     def test_parameter_count(self, gat_model):
#         num_params = sum(p.numel() for p in gat_model.parameters())
#         assert num_params > 0, "Model should have parameters"
#         assert num_params < 1e7, "Model should not have an excessive number of parameters"

#     def test_deterministic_forward(self, model_config, seed_fixture):
#         seed_fixture(42)
#         gat = GraphAttentionNetwork(model_config)
#         gat.eval()

#         batch_size = 2
#         num_nodes = 10
#         x = torch.randn(batch_size, num_nodes, model_config.node_features)
#         adj = torch.randint(0, 2, (batch_size, num_nodes, num_nodes)).float()

#         with torch.no_grad():
#             out1, _, _ = gat(x, adj)
#             out2, _, _ = gat(x, adj)
#             assert torch.allclose(out1, out2, rtol=1e-6)



# import pytest
# import torch 
# from NeuroFusion.models.gat import GraphAttentionNetwork, GraphAttentionLayer, GatedGraphPooling

class TestGraphAttentionNetwork:
    """Tests for Graph Attention Network"""
    def test_forward_shape(self, model_config, sample_node_features, sample_adj_matrix):
        gat = GraphAttentionNetwork(model_config)

        # Unpacking order: (node_output, intermediates, pooled_graph_output)
        output, intermediate_features, pooled_output = gat(sample_node_features, sample_adj_matrix)
  
        batch_size, num_nodes, _ = sample_node_features.shape 

        assert output.shape == (batch_size, num_nodes, model_config.gat_out_dim)
        assert pooled_output.shape == (batch_size, model_config.fusion_dim)
        assert len(intermediate_features) == model_config.gat_num_layers
        

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