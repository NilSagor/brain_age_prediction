# src/NeuroFusion/lightning_module/neurofusion_lightning.py

import torch
import torch.nn.functional as F
import lightning as L
from typing import Dict, Optional, Any, List
from ..models.neurofusion import NeuroFusion
from NeuroFusion.config import ModelConfig, TrainingConfig


class NeuroFusionLightning(L.LightningModule):
    """
    PyTorch Lightning wrapper for the NeuroFusion core model.
    """
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig):
        super().__init__()
        self.save_hyperparameters()
        self.model_config = model_config
        self.training_config = training_config
        
        # Instantiate the core model
        self.model = NeuroFusion(model_config)
        
        # For epoch‑level aggregation
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def forward(self, sMRI: torch.Tensor, fMRI: torch.Tensor,
                adj: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass – delegates to core model."""
        return self.model(sMRI, fMRI, adj, return_all=True)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        # batch must contain 'sMRI', 'fMRI', 'age', and optionally 'adj'
        loss_dict = self.model.compute_loss(batch, self.training_config)
        
        # Log losses
        self.log('train_loss', loss_dict['loss'], prog_bar=True)
        self.log('train_main_loss', loss_dict['main_loss'])
        for i, aux_loss in enumerate(loss_dict['aux_losses']):
            self.log(f'train_aux_loss_{i}', aux_loss)
        
        # Log MAE
        with torch.no_grad():
            mae = F.l1_loss(loss_dict['final_prediction'], batch['age'])
            self.log('train_mae', mae, prog_bar=True)
        
        return loss_dict['loss']
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step."""
        loss_dict = self.model.compute_loss(batch, self.training_config)
        
        self.log('val_loss', loss_dict['loss'], prog_bar=True)
        self.log('val_main_loss', loss_dict['main_loss'])
        for i, aux_loss in enumerate(loss_dict['aux_losses']):
            self.log(f'val_aux_loss_{i}', aux_loss)
        
        mae = F.l1_loss(loss_dict['final_prediction'], batch['age'])
        self.log('val_mae', mae, prog_bar=True)
        
        # Store for epoch‑end aggregation
        self.validation_step_outputs.append({
            'preds': loss_dict['final_prediction'],
            'targets': batch['age']
        })
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Test step."""
        loss_dict = self.model.compute_loss(batch, self.training_config)
        
        self.log('test_loss', loss_dict['loss'])
        mae = F.l1_loss(loss_dict['final_prediction'], batch['age'])
        mse = F.mse_loss(loss_dict['final_prediction'], batch['age'])
        self.log('test_mae', mae)
        self.log('test_mse', mse)
        
        self.test_step_outputs.append({
            'preds': loss_dict['final_prediction'],
            'targets': batch['age']
        })
    
    def on_validation_epoch_end(self) -> None:
        """Aggregate validation metrics at epoch end."""
        if not self.validation_step_outputs:
            return
        preds = torch.cat([out['preds'] for out in self.validation_step_outputs])
        targets = torch.cat([out['targets'] for out in self.validation_step_outputs])
        
        mae = F.l1_loss(preds, targets)
        mse = F.mse_loss(preds, targets)
        r = self._compute_correlation(preds, targets)
        
        self.log('val_epoch_mae', mae, prog_bar=True)
        self.log('val_epoch_mse', mse)
        self.log('val_epoch_corr', r)
        
        self.validation_step_outputs.clear()
    
    def on_test_epoch_end(self) -> None:
        """Aggregate test metrics at epoch end."""
        if not self.test_step_outputs:
            return
        preds = torch.cat([out['preds'] for out in self.test_step_outputs])
        targets = torch.cat([out['targets'] for out in self.test_step_outputs])
        
        mae = F.l1_loss(preds, targets)
        mse = F.mse_loss(preds, targets)
        r = self._compute_correlation(preds, targets)
        
        self.log('test_final_mae', mae)
        self.log('test_final_mse', mse)
        self.log('test_final_corr', r)
        
        self.test_step_outputs.clear()
    
    @staticmethod
    def _compute_correlation(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Pearson correlation coefficient."""
        preds_mean = preds.mean()
        targets_mean = targets.mean()
        num = ((preds - preds_mean) * (targets - targets_mean)).sum()
        den = torch.sqrt(((preds - preds_mean).pow(2).sum() * (targets - targets_mean).pow(2).sum()))
        return num / (den + 1e-8)
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.training_config.num_epochs,
            eta_min=1e-6
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }




# class RegressionHead(nn.Module):
#     """Regression head for brain age estimation."""
#     def __init__(self, in_dim: int = 768, hidden_dim: int = 256, dropout: float = 0.2):
#         super().__init__()
#         self.fc1 = nn.Linear(in_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, 1)
#         self.gelu = nn.GELU()
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.fc1(x)
#         x = self.gelu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x.squeeze(-1)