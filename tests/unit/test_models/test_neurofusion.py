# tests/unit/test_models/test_neurofusion.py
import pytest
import torch
import torch.nn as nn
from NeuroFusion.models.neurofusion import NeuroFusion


class TestNeuroFusion:
    """Complete tests for NeuroFusion model"""
    
    def test_forward_shape(self, model_config, neurofusion_model):
        """Test output shapes from NeuroFusion"""
        batch_size = 2
        sMRI = torch.randn(batch_size, 1, model_config.img_size,
                          model_config.img_size, model_config.img_size)
        fMRI = torch.randn(batch_size, model_config.num_nodes, model_config.node_features)
        
        output = neurofusion_model(sMRI, fMRI, return_all=True)
        
        # Check final prediction
        assert 'final_prediction' in output
        assert output['final_prediction'].shape == (batch_size,)
        
        # Check aux predictions
        assert 'aux_predictions' in output
        assert len(output['aux_predictions']) == len(model_config.structural_stages)
        for aux_pred in output['aux_predictions']:
            assert aux_pred.shape == (batch_size,)
        
        # Check fused features
        assert 'fused_features' in output
        assert len(output['fused_features']) == len(model_config.structural_stages)
        for fused in output['fused_features']:
            assert fused.shape == (batch_size, model_config.fusion_dim)
    
    def test_forward_with_adjacency(self, model_config, neurofusion_model):
        """Test forward pass with adjacency matrix"""
        batch_size = 2
        sMRI = torch.randn(batch_size, 1, model_config.img_size,
                          model_config.img_size, model_config.img_size)
        fMRI = torch.randn(batch_size, model_config.num_nodes, model_config.node_features)
        adj = torch.randint(0, 2, (batch_size, model_config.num_nodes, model_config.num_nodes)).float()
        
        output = neurofusion_model(sMRI, fMRI, adj, return_all=True)
        
        assert output['final_prediction'].shape == (batch_size,)
    
    def test_inference_mode(self, model_config, neurofusion_model):
        """Test inference mode (return_all=False)"""
        batch_size = 2
        sMRI = torch.randn(batch_size, 1, model_config.img_size,
                          model_config.img_size, model_config.img_size)
        fMRI = torch.randn(batch_size, model_config.num_nodes, model_config.node_features)
        
        output = neurofusion_model(sMRI, fMRI, return_all=False)
        
        assert 'final_prediction' in output
        assert output['final_prediction'].shape == (batch_size,)
        assert 'aux_predictions' not in output
    
    def test_compute_loss(self, model_config, neurofusion_model, batch_data, training_config):
        """Test loss computation"""
        batch = batch_data()
        
        loss_dict = neurofusion_model.compute_loss(batch, training_config)
        
        assert 'loss' in loss_dict
        assert 'main_loss' in loss_dict
        assert 'aux_losses' in loss_dict
        assert len(loss_dict['aux_losses']) == len(model_config.structural_stages)
        
        # Check that losses are scalar
        for loss in [loss_dict['loss'], loss_dict['main_loss']] + loss_dict['aux_losses']:
            assert loss.numel() == 1
            assert torch.isfinite(loss)
    
    def test_gradient_flow(self, model_config, neurofusion_model, batch_data, training_config):
        """Test gradient flow through entire model"""
        batch = batch_data()
        
        # Forward
        loss_dict = neurofusion_model.compute_loss(batch, training_config)
        loss = loss_dict['loss']
        
        # Backward
        loss.backward()
        
        # Check that all parameters have gradients
        for name, param in neurofusion_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"
    
    def test_parameter_count(self, model_config, neurofusion_model):
        """Test total parameter count is reasonable"""
        total_params = sum(p.numel() for p in neurofusion_model.parameters())
        trainable_params = sum(p.numel() for p in neurofusion_model.parameters() if p.requires_grad)
        
        # For test config, should be manageable
        assert total_params > 0
        assert total_params < 30_000_000
        assert trainable_params == total_params  # All parameters should be trainable
    
    def test_deterministic_forward(self, model_config, neurofusion_model):
        """Test deterministic forward pass"""
        torch.manual_seed(42)
        model = NeuroFusion(model_config)
        model.eval()
        
        batch_size = 2
        sMRI = torch.randn(batch_size, 1, model_config.img_size,
                          model_config.img_size, model_config.img_size)
        fMRI = torch.randn(batch_size, model_config.num_nodes, model_config.node_features)
        
        with torch.no_grad():
            output1 = model(sMRI, fMRI, return_all=False)
            output2 = model(sMRI, fMRI, return_all=False)
            
            assert torch.allclose(output1['final_prediction'], output2['final_prediction'], rtol=1e-6)
    
    def test_hierarchical_supervision(self, model_config, neurofusion_model, batch_data):
        """Test that hierarchical supervision affects intermediate representations"""
        batch = batch_data()
        
        # Forward pass
        output = neurofusion_model(batch['sMRI'], batch['fMRI'], return_all=True)
        
        # Check that aux predictions are different and meaningful
        aux_preds = output['aux_predictions']
        for i, pred in enumerate(aux_preds):
            assert torch.isfinite(pred).all()
            # Predictions should be in reasonable range
            assert pred.min() > -200, f"Aux prediction {i} has unexpectedly negative values"
            assert pred.max() < 200, f"Aux prediction {i} has unexpectedly large values"
    
    def test_fusion_gates_variation(self, model_config, neurofusion_model, batch_data):
        """Test that fusion gates vary across stages"""
        batch = batch_data()
        
        output = neurofusion_model(batch['sMRI'], batch['fMRI'], return_all=True)
        gates = output['gates']
        
        # Check that gates are different across stages
        if len(gates) > 1:
            for i in range(len(gates) - 1):
                # Gates should not be identical
                assert not torch.allclose(gates[i], gates[i+1], rtol=1e-3), f"Gates {i} and {i+1} are identical"
    
    def test_model_save_load(self, model_config, neurofusion_model, tmp_path):
        """Test model saving and loading"""
        # Save
        save_path = tmp_path / "model.pth"
        torch.save(neurofusion_model.state_dict(), save_path)
        
        # Create new model and load
        new_model = NeuroFusion(model_config)
        new_model.load_state_dict(torch.load(save_path))
        
        # Compare outputs
        batch_size = 2
        sMRI = torch.randn(batch_size, 1, model_config.img_size,
                          model_config.img_size, model_config.img_size)
        fMRI = torch.randn(batch_size, model_config.num_nodes, model_config.node_features)
        
        neurofusion_model.eval()
        new_model.eval()
        
        with torch.no_grad():
            output1 = neurofusion_model(sMRI, fMRI, return_all=False)
            output2 = new_model(sMRI, fMRI, return_all=False)
            
            assert torch.allclose(output1['final_prediction'], output2['final_prediction'], rtol=1e-5)