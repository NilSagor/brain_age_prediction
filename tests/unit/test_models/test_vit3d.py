# tests/test_vit3d.py
import pytest
import torch 


from NeuroFusion.models.vit3d import (
    PatchEmbed3D,
    Attention3D,
    Mlp3D,
    Block3D,
    VisionTransformer3D
)

class TestPatchEmbed3D:
    """Tests for 3D Patch Embedding"""
    def test_forward_shape(self, model_config):
        patch_embed = PatchEmbed3D(
            img_size = model_config.img_size,
            patch_size = model_config.patch_size,
            in_chans = model_config.in_channels,
            embed_dim = model_config.embed_dim
        )

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