# tests/test_vit3d.py
from NeuroFusion.models.vit3d import Attention3D
from black import output
import pytest
import torch

from tests.conftest import get_attention3D 




class TestPatchEmbed3D:
    """Tests for 3D Patch Embedding"""
    def test_forward_shape(self, model_config, get_patchEmbed3D):
        patch_embed = get_patchEmbed3D

        batch_size = 2
        x = torch.randn(
            batch_size, 
            1, 
            model_config.img_size, 
            model_config.img_size, 
            model_config.img_size, 
            )
        output = patch_embed(x)
        expected_patches = (model_config.img_size//model_config.patch_size)**3
        assert output.shape == (batch_size, expected_patches, model_config.embed_dim)

        batch_size = 2
        x = torch.randn(
            batch_size, 
            1, 
            model_config.img_size, 
            model_config.img_size, 
            model_config.img_size, 
            )
        output = patch_embed(x)
        expected_patches = (model_config.img_size//model_config.patch_size)**3
        assert output.shape == (batch_size, expected_patches, model_config.embed_dim)

class TestAttention3D:
    """Tests for 3D Multi-Head Self Attention"""
    
    def test_forward_shape(self, model_config, get_attention3D):
        """Test output shape of attention module"""
        attn = get_attention3D
        
        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, model_config.embed_dim)
        output = attn(x)
        
        assert output.shape == (batch_size, seq_len, model_config.embed_dim)
    
    def test_attention_weights(self, model_config, get_attention3D):
        """Test that attention weights sum to 1"""
        attn = get_attention3D
        attn.eval()  
        
        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, model_config.embed_dim)
        
        with torch.no_grad():
            # Access attention weights through forward
            B, N, C = x.shape
            qkv = attn.qkv(x).reshape(B, N, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn_weights = (q @ k.transpose(-2, -1)) * attn.scale
            attn_weights = attn_weights.softmax(dim=-1)
            
            # Check that each row sums to 1
            row_sums = attn_weights.sum(dim=-1)
            assert torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-5)




class TestVisionTransformer3D:
    """Tests for 3D Vision Transformer"""
    def test_forward_shape(self, model_config, get_vit_model):
        """Test output shapes from ViT"""
        batch_size = 2
        x = torch.randn(batch_size, 1, model_config.img_size,
                       model_config.img_size, model_config.img_size)
        
        vit_model = get_vit_model
        pooled, intermediate, tokens = vit_model(x)
        
        # Check shapes
        assert pooled.shape == (batch_size, model_config.embed_dim)
        assert tokens.shape == (batch_size, vit_model.num_patches, model_config.embed_dim)
        
        # Check intermediate features
        expected_stages = len(model_config.structural_stages)
        assert len(intermediate) == expected_stages
        for feat in intermediate:
            assert feat.shape == (batch_size, vit_model.num_patches, model_config.embed_dim)
    
    def test_intermediate_feature_extraction(self, model_config, get_vit_model):
        """Test that intermediate features are extracted at correct layers"""
        vit = get_vit_model
        x = torch.randn(2, 1, model_config.img_size,
                       model_config.img_size, model_config.img_size)
        
        _, intermediate, _ = vit(x, return_intermediate=True)
        
        # Should extract at specified layers
        assert len(intermediate) == len(model_config.structural_stages)
    
    def test_gradient_flow(self, model_config, get_vit_model):
        """Test gradient flow through entire ViT"""
        vit = get_vit_model
        x = torch.randn(2, 1, model_config.img_size,
                       model_config.img_size, model_config.img_size,
                       requires_grad=True)
        
        pooled, _, _ = vit(x)
        loss = pooled.sum()
        loss.backward()
        
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        
        # Check parameter gradients
        for name, param in vit.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"
    
    def test_parameter_count(self, model_config, get_vit_model):
        """Test parameter count is reasonable"""
        vit = get_vit_model
        num_params = sum(p.numel() for p in vit.parameters())
        
        # For reduced config, should have reasonable number
        assert num_params > 0
        assert num_params < 50_000_000  # Should be less than 50M for test config
    
    def test_deterministic_forward(self, model_config, get_vit_model):
        """Test that forward pass is deterministic with fixed seed"""
        torch.manual_seed(42)
        vit = get_vit_model
        vit.eval()
        
        x = torch.randn(2, 1, model_config.img_size,
                       model_config.img_size, model_config.img_size)
        
        with torch.no_grad():
            output1 = vit(x)
            output2 = vit(x)
            
            # Check both pooled and token outputs
            for o1, o2 in zip(output1, output2):
                if isinstance(o1, torch.Tensor) and isinstance(o2, torch.Tensor):
                    assert torch.allclose(o1, o2, rtol=1e-6)

