from pathlib import Path

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from CLAM.datasets.dataset_generic import Generic_MIL_Dataset


def collate_fn(batches):
    features = [batch[0] for batch in batches]
    labels = torch.tensor([torch.tensor(batch[1]) for batch in batches])
    coords = [batch[2] for batch in batches]
    return features, labels, coords


class MILDatamodule(LightningDataModule):
    # Inspired by https://github.com/gokul-pv/lightning-hydra-timm
    def __init__(
        self,
        dataset: Generic_MIL_Dataset,
        split_csv_filename: Path,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = False,
        shuffle: bool = True,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.split_csv_file = split_csv_filename
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.data_train, self.data_val, self.data_test = self.dataset.return_splits(
            from_id=False,
            csv_path=self.split_csv_file,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=collate_fn,
        )
