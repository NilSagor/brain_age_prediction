import warnings

import numpy as np
import torch
from nilearn.datasets import fetch_abide_pcp
from torch.utils.data import Dataset, random_split

warnings.filterwarnings("ignore", category=RuntimeWarning)


class ABIDEDataset(Dataset):
    """
    ABIDE Dataset for brain age estimation development.

    This dataset uses nilearn to fetch preprocessed resting-state fMRI
    time series (rois_cc200) for typical controls (TDC) and computes
    functional connectivity matrices on the fly.

    Args:
        data_dir: Directory to store downloaded data (default: './data/abide_data')
        age_min: Minimum age (default: 18)
        age_max: Maximum age (default: 64)
        roi_atlas: ROI atlas, e.g., 'cc200', 'cc400', or 'ho' (Harvard-Oxford)
        quality_checked: Use only quality‑checked subjects (recommended)
        band_pass: Apply band‑pass filtering
        global_signal: Apply global signal regression
        transform: Optional transform applied to items
    """

    def __init__(
        self,
        data_dir: str = "./data/abide_data",
        age_min: int = 18,
        age_max: int = 64,
        roi_atlas: str = "cc200",
        quality_checked: bool = True,
        band_pass: bool = True,
        global_signal: bool = True,
        transform: callable | None = None,
    ):
        self.data_dir = data_dir
        self.age_min = age_min
        self.age_max = age_max
        self.roi_atlas = roi_atlas
        self.quality_checked = quality_checked
        self.band_pass = band_pass
        self.global_signal = global_signal
        self.transform = transform

        # Fetch the data
        self._fetch_data()

    def _fetch_data(self):
        """Download or load cached ABIDE data using nilearn."""
        # Fetch all subjects (no diagnosis filter)
        self.abide_data = fetch_abide_pcp(
            data_dir=self.data_dir,
            pipeline="cpac",
            band_pass_filtering=self.band_pass,
            global_signal_regression=self.global_signal,
            derivatives=[f"rois_{self.roi_atlas}"],
            quality_checked=self.quality_checked,
        )

        # Phenotypic DataFrame
        pheno = self.abide_data.phenotypic

        # Filter: typical controls (DX_GROUP == 1)
        tdc_mask = pheno["DX_GROUP"] == 1

        # Filter: age range
        age_mask = (pheno["AGE_AT_SCAN"] >= self.age_min) & (pheno["AGE_AT_SCAN"] <= self.age_max)

        final_mask = tdc_mask & age_mask

        # Apply masks
        pheno_filtered = pheno[final_mask].reset_index(drop=True)
        roi_series = self.abide_data.rois_cc200
        roi_filtered = [roi_series[i] for i in range(len(roi_series)) if final_mask.iloc[i]]

        self.phenotypic = pheno_filtered
        self.roi_time_series = roi_filtered

        print(
            f"Retained {len(self.roi_time_series)} typical controls aged {self.age_min}–{self.age_max}."
        )

    def __len__(self):
        return len(self.roi_time_series)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Get ROI time series (T x R) – T timepoints, R regions
        ts = self.roi_time_series[idx]  # numpy array

        # Compute correlation matrix (R x R)
        corr = np.corrcoef(ts.T)
        corr = np.nan_to_num(corr)  # handle NaNs

        # Age
        age = float(self.phenotypic.iloc[idx]["AGE_AT_SCAN"])

        # Return as dictionary
        item = {
            "age": torch.tensor(age, dtype=torch.float32),
            "fMRI": torch.from_numpy(corr).float(),
            "sub_id": str(self.phenotypic.iloc[idx]["SUB_ID"]),
            "site": str(self.phenotypic.iloc[idx]["SITE_ID"]),
        }

        if self.transform:
            item = self.transform(item)

        return item


class ABIDEDataModule:
    """
    PyTorch Lightning DataModule for ABIDE.
    Similar to CamCANDataModule, but for ABIDE.
    """

    def __init__(
        self,
        data_dir: str = "./data/abide_data",
        batch_size: int = 8,
        num_workers: int = 4,
        age_min: int = 18,
        age_max: int = 64,
        roi_atlas: str = "cc200",
        quality_checked: bool = True,
        val_split: float = 0.1,
        test_split: float = 0.15,
        seed: int = 42,
        **kwargs,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.age_min = age_min
        self.age_max = age_max
        self.roi_atlas = roi_atlas
        self.quality_checked = quality_checked
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.kwargs = kwargs

        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):

        self.dataset = ABIDEDataset(
            data_dir=self.data_dir,
            age_min=self.age_min,
            age_max=self.age_max,
            roi_atlas=self.roi_atlas,
            quality_checked=self.quality_checked,
            **self.kwargs,
        )

        n = len(self.dataset)
        n_train = int(n * (1 - self.val_split - self.test_split))
        n_val = int(n * self.val_split)
        n_test = n - n_train - n_val

        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [n_train, n_val, n_test], generator=generator
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
