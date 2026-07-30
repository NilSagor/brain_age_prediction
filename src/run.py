from NeuroFusion.data_prep.abide_dataset import ABIDEDataModule

# For development/validation on ABIDE
data_module = ABIDEDataModule(
    data_dir="./data/abide_data",
    batch_size=8,
    age_min=18,
    age_max=64,
    roi_atlas="cc200",
    quality_checked=True,
)

data_module.setup()
train_loader = data_module.train_dataloader()
batch = next(iter(train_loader))
print(batch.keys())
