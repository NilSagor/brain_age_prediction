# tests/unit/test_training/test_neurofusion_lightning.py
import pytest
import torch
import lightning as L
from torch.utils.data import DataLoader, Dataset
from NeuroFusion.lightning_module.neurofusion_lightning import NeuroFusionLightning
from NeuroFusion.config import ModelConfig, TrainingConfig

from lightning.pytorch.trainer.states import TrainerStatus

class DummyDataset(Dataset):
    """A minimal dataset that yields batches matching the expected format."""
    def __init__(self, num_samples=20, config=None):
        self.num_samples = num_samples
        self.config = config or ModelConfig()
        self.img_size = self.config.img_size
        self.num_nodes = self.config.num_nodes
        self.node_features = self.config.node_features

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sMRI = torch.randn(1, self.img_size, self.img_size, self.img_size)
        fMRI = torch.randn(self.num_nodes, self.node_features)
        age = torch.rand(1).item() * 60 + 20  # ages 20-80
        adj = torch.randint(0, 2, (self.num_nodes, self.num_nodes)).float()
        return {
            'sMRI': sMRI,
            'fMRI': fMRI,
            'age': torch.tensor(age, dtype=torch.float32),
            'adj': adj
        }


class DummyDataModule(L.LightningDataModule):
    def __init__(self, config, batch_size=2):
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.dataset = DummyDataset(config=config)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)


class TestNeuroFusionLightning:
    """Tests for the Lightning wrapper."""

    def test_initialization(self, model_config, training_config):
        """Check that the module saves hyperparameters and instantiates the core model."""
        module = NeuroFusionLightning(model_config, training_config)
        assert module.model is not None
        assert hasattr(module, 'hparams')
        assert module.model_config == model_config
        assert module.training_config == training_config
        assert module.validation_step_outputs == []
        assert module.test_step_outputs == []

    def test_forward_pass(self, model_config, training_config):
        """Test that forward returns the expected dict with final_prediction."""
        module = NeuroFusionLightning(model_config, training_config)
        batch_size = 2
        sMRI = torch.randn(batch_size, 1, model_config.img_size,
                           model_config.img_size, model_config.img_size)
        fMRI = torch.randn(batch_size, model_config.num_nodes,
                           model_config.node_features)
        adj = torch.randint(0, 2, (batch_size, model_config.num_nodes,
                                   model_config.num_nodes)).float()

        # Forward in training mode (returns all)
        outputs = module(sMRI, fMRI, adj)
        assert 'final_prediction' in outputs
        assert outputs['final_prediction'].shape == (batch_size,)
        # Also check that other keys (aux_predictions, etc.) are present if core returns them
        assert 'aux_predictions' in outputs

    def test_training_step(self, model_config, training_config, batch_data):
        """Test that training_step computes a scalar loss and logs metrics."""
        module = NeuroFusionLightning(model_config, training_config)
        batch = batch_data(batch_size=2)
        loss = module.training_step(batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1
        assert torch.isfinite(loss)

        # Check that logs were recorded (we can use the logger mock, but we'll just check
        # that the loss value is not None; logging is tested by Lightning itself).

    def test_validation_step(self, model_config, training_config, batch_data):
        """Test that validation_step stores outputs and logs val_loss."""
        module = NeuroFusionLightning(model_config, training_config)
        batch = batch_data(batch_size=2)
        # Clear any previous outputs
        module.validation_step_outputs = []
        module.validation_step(batch, batch_idx=0)
        assert len(module.validation_step_outputs) == 1
        out = module.validation_step_outputs[0]
        assert 'preds' in out
        assert 'targets' in out
        assert out['preds'].shape == out['targets'].shape

    def test_test_step(self, model_config, training_config, batch_data):
        """Test that test_step stores outputs and logs test metrics."""
        module = NeuroFusionLightning(model_config, training_config)
        batch = batch_data(batch_size=2)
        module.test_step_outputs = []
        module.test_step(batch, batch_idx=0)
        assert len(module.test_step_outputs) == 1
        out = module.test_step_outputs[0]
        assert 'preds' in out
        assert 'targets' in out

    def test_validation_epoch_end(self, model_config, training_config, batch_data):
        """Test that epoch-end aggregation computes MAE, MSE, correlation correctly."""
        module = NeuroFusionLightning(model_config, training_config)
        # Simulate two validation steps
        batch1 = batch_data(batch_size=2)
        batch2 = batch_data(batch_size=3)
        module.validation_step(batch1, 0)
        module.validation_step(batch2, 1)
        # Now call epoch end
        module.on_validation_epoch_end()
        # After clearing, the list should be empty
        assert module.validation_step_outputs == []
        # We can't easily check the logged values without a logger mock,
        # but we can check that the method didn't crash.
        # In a real test, we'd use a logger mock to verify calls.

    def test_test_epoch_end(self, model_config, training_config, batch_data):
        """Test test epoch end aggregation."""
        module = NeuroFusionLightning(model_config, training_config)
        batch1 = batch_data(batch_size=2)
        batch2 = batch_data(batch_size=3)
        module.test_step(batch1, 0)
        module.test_step(batch2, 1)
        module.on_test_epoch_end()
        assert module.test_step_outputs == []

    def test_configure_optimizers(self, model_config, training_config):
        """Test that optimizer and scheduler are correctly configured."""
        module = NeuroFusionLightning(model_config, training_config)
        opt_config = module.configure_optimizers()
        assert 'optimizer' in opt_config
        assert 'lr_scheduler' in opt_config
        optimizer = opt_config['optimizer']
        scheduler = opt_config['lr_scheduler']['scheduler']
        assert isinstance(optimizer, torch.optim.AdamW)
        # Check that the scheduler is a CosineAnnealingLR
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_training_loop_integration(self, model_config, training_config):
        """Integration test: run a full training loop with a dummy datamodule."""
        # Use small config for speed
        config = model_config
        # Override with tiny values for fast test
        config.img_size = 16
        config.num_nodes = 4
        config.node_features = 4
        config.embed_dim = 16
        config.gat_hidden_dim = 8
        config.fusion_dim = 8
        config.num_layers = 2
        config.gat_num_layers = 2
        config.structural_stages = [1, 2]
        config.functional_stages = [1, 2]
        config.aux_weights = [0.3, 0.5]

        training_config.num_epochs = 1

        module = NeuroFusionLightning(config, training_config)
        datamodule = DummyDataModule(config, batch_size=2)

        # Use a fast_dev_run to validate the loop quickly
        trainer = L.Trainer(
            max_epochs=1,
            accelerator='cpu',
            devices=1,
            fast_dev_run=True,  # runs only one batch per epoch
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(module, datamodule)
        
        assert trainer.state.status == TrainerStatus.FINISHED

    def test_gradient_flow_in_lightning(self, model_config, training_config, batch_data):
        """Test that gradients flow through the Lightning module during training."""
        module = NeuroFusionLightning(model_config, training_config)
        batch = batch_data(batch_size=2)

        # Simulate a training step and backward
        loss = module.training_step(batch, 0)
        loss.backward()

        # Check that all parameters have gradients
        for name, param in module.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"
                assert torch.isfinite(param.grad).all(), f"Gradient for {name} contains NaN/Inf"