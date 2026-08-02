# tests/test_fusion.py
import pytest
import torch



class TestMultiHeadCrossAttention:
    """Tests for Multi-Head Cross Attention"""
    def test_forward_shape(self, model_config, get_multihead_cross_attention):
        cross_attention = get_multihead_cross_attention        

        batch_size = 2
        num_nodes_structural = 10
        num_nodes_functional = 15

        structural = torch.randn(batch_size, num_nodes_structural, model_config.embed_dim)
        functional = torch.randn(batch_size, num_nodes_functional, model_config.embed_dim)

        output, pooled = cross_attention(structural, functional)
        assert output.shape == (batch_size, num_nodes_structural, model_config.embed_dim)
        assert pooled is None or pooled.shape == (batch_size, 1, model_config.embed_dim)

    def test_gradient_flow(self, model_config, get_multihead_cross_attention):
        cross_attention = get_multihead_cross_attention

        batch_size = 2
        N_q = 5
        N_kv = 7
        structural = torch.randn(batch_size, N_q, model_config.embed_dim, requires_grad=True)
        functional = torch.randn(batch_size, N_kv, model_config.embed_dim, requires_grad=True)

        output, _ = cross_attention(structural, functional)
        loss = output.sum()
        loss.backward()

        assert structural.grad is not None
        assert functional.grad is not None
        assert torch.isfinite(structural.grad).all()
        assert torch.isfinite(functional.grad).all()

    def test_with_pooled_input(self, model_config, get_multihead_cross_attention):
        cross_attention = get_multihead_cross_attention

        batch_size = 2
        structural = torch.randn(batch_size, model_config.embed_dim)  # pooled, shape (B, D)
        functional = torch.randn(batch_size, model_config.embed_dim)

        output, pooled = cross_attention(structural, functional)
        assert output.shape == (batch_size, 1, model_config.embed_dim)  # output is (B, 1, D)
        assert pooled.shape == (batch_size, 1, model_config.embed_dim)

    def test_with_mask(self, model_config, get_multihead_cross_attention):
        cross_attention = get_multihead_cross_attention

        batch_size = 2
        N_q = 5
        N_kv = 7
        structural = torch.randn(batch_size, N_q, model_config.embed_dim)
        functional = torch.randn(batch_size, N_kv, model_config.embed_dim)

        # Create a mask that allows only first 3 keys
        mask = torch.zeros(batch_size, N_q, N_kv, dtype=torch.bool)
        mask[:, :, :3] = True

        output, _ = cross_attention(structural, functional, mask=mask)
        assert output.shape == (batch_size, N_q, model_config.embed_dim)

    # def test_with_token_features(self, model_config, get_multihead_cross_attention):
    #     cross_attention = get_multihead_cross_attention


    #     batch_size = 2
    #     num_tokens = 8  # e.g., [CLS] token
    #     num_nodes_structural = 10
    #     num_nodes_functional = 15

    #     structural = torch.randn(batch_size, num_tokens, model_config.embed_dim)
    #     functional = torch.randn(batch_size, num_tokens, model_config.embed_dim)

    #     # Add a token feature (e.g., [CLS] token) to the structural input
    #     cls_token = torch.randn(batch_size, num_tokens, model_config.embed_dim)
    #     structural_with_token = torch.cat([cls_token, structural], dim=1)

    #     output, pooled = cross_attention(structural_with_token, functional)
    #     assert output.shape == (batch_size, num_tokens + num_nodes_structural, model_config.embed_dim)
    #     assert pooled is None or pooled.shape == (batch_size, 1, model_config.embed_dim)    


    #     assert output.shape == (batch_size, num_tokens, model_config.embed_dim)


class TestGatedFusion:
    """Tests for Gated Fusion"""
    def test_forward_shape(self, model_config, get_gated_fusion):
        gated_fusion = get_gated_fusion

        batch_size = 2
        num_nodes = 10

        structural = torch.randn(batch_size, num_nodes, model_config.embed_dim)
        functional = torch.randn(batch_size, num_nodes, model_config.embed_dim)

        fused_output, gate_values = gated_fusion(structural, functional)
        assert fused_output.shape == (batch_size, num_nodes, model_config.embed_dim)
        assert gate_values.shape == (batch_size, num_nodes, model_config.embed_dim)

    def test_gate_range(self, model_config, get_gated_fusion):
        gated_fusion = get_gated_fusion

        batch_size = 2
        num_nodes = 10

        structural = torch.randn(batch_size, num_nodes, model_config.embed_dim)
        functional = torch.randn(batch_size, num_nodes, model_config.embed_dim)

        gated_fusion.eval()
        _, gate_values = gated_fusion(structural, functional)
        assert torch.all(gate_values >= 0) and torch.all(gate_values <= 1), "Gate values should be in the range [0, 1]"

    def test_adaptive_weighting(self, model_config, get_gated_fusion):
        gated_fusion = get_gated_fusion

        batch_size = 2
        num_nodes = 10

        structural1 = torch.ones(batch_size, num_nodes, model_config.embed_dim)
        functional1 = torch.ones(batch_size, num_nodes, model_config.embed_dim)

        structural2 = torch.zeros(batch_size, num_nodes, model_config.embed_dim)
        functional2 = torch.zeros(batch_size, num_nodes, model_config.embed_dim)

        gated_fusion.eval()
        _, gate_values1 = gated_fusion(structural1, functional1)
        _, gate_values2 = gated_fusion(structural2, functional2)

        assert not torch.allclose(gate_values1, gate_values2, rtol=1e-5), "Gate values should adapt based on input features"

    def test_gradient_flow(self, model_config, get_gated_fusion):
        gated_fusion = get_gated_fusion

        batch_size = 2
        num_nodes = 10

        structural = torch.randn(batch_size, num_nodes, model_config.embed_dim, requires_grad=True)
        functional = torch.randn(batch_size, num_nodes, model_config.embed_dim, requires_grad=True)

        fused_output, _ = gated_fusion(structural, functional)
        loss = fused_output.sum()
        loss.backward()

        assert structural.grad is not None, "Gradients should flow back to the structural input"
        assert functional.grad is not None, "Gradients should flow back to the functional input"
        assert torch.isfinite(structural.grad).all(), "Gradient contains NaN or Inf values" 
        assert torch.isfinite(functional.grad).all(), "Gradient contains NaN or Inf values"

    def test_identity_property(self, model_config, get_gated_fusion):
        gated_fusion = get_gated_fusion

        batch_size = 2
        num_nodes = 10

        structural = torch.randn(batch_size, num_nodes, model_config.embed_dim)
        functional = torch.randn(batch_size, num_nodes, model_config.embed_dim)

        # Test when structural and functional are the same
        fused_output_same, gate_values_same = gated_fusion(structural, structural)
        assert torch.allclose(fused_output_same, structural, rtol=1e-5), "Fused output should be close to the input when both inputs are the same"

class TestHierarchicalFusionBlock:
    """Tests for Hierarchical Fusion Block"""
    def test_forward_shape(self, model_config, get_hierarchical_fusion_block):
        fusion_block = get_hierarchical_fusion_block

        batch_size = 2
        num_nodes = 10

        structural = torch.randn(batch_size, num_nodes, model_config.embed_dim)
        functional = torch.randn(batch_size, num_nodes, model_config.embed_dim)

        fused_output, gate_values = fusion_block(structural, functional)
        assert fused_output.shape == (batch_size, num_nodes, model_config.embed_dim)
        assert gate_values.shape == (batch_size, num_nodes, model_config.embed_dim)

    def test_gradient_flow(self, model_config, get_hierarchical_fusion_block):
        fusion_block = get_hierarchical_fusion_block

        batch_size = 2
        num_nodes = 10

        structural = torch.randn(batch_size, num_nodes, model_config.embed_dim, requires_grad=True)
        functional = torch.randn(batch_size, num_nodes, model_config.embed_dim, requires_grad=True)

        fused_output, _ = fusion_block(structural, functional)
        loss = fused_output.sum()
        loss.backward()

        assert structural.grad is not None, "Gradients should flow back to the structural input"
        assert functional.grad is not None, "Gradients should flow back to the functional input"
        assert torch.isfinite(structural.grad).all(), "Gradient contains NaN or Inf values" 
        assert torch.isfinite(functional.grad).all(), "Gradient contains NaN or Inf values"

    def test_layer_norm(self, model_config, get_hierarchical_fusion_block):
        fusion_block = get_hierarchical_fusion_block

        batch_size = 2
        num_nodes = 10

        structural = torch.randn(batch_size, num_nodes, model_config.embed_dim)
        functional = torch.randn(batch_size, num_nodes, model_config.embed_dim)

        fused_output, _ = fusion_block(structural, functional)
        # Check if the output has zero mean and unit variance (approximately)
        mean = fused_output.mean(dim=-1)
        std = fused_output.std(dim=-1)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-2), "Output mean should be close to zero"
        assert torch.allclose(std, torch.ones_like(std), atol=1e-1), "Output std should be close to one"