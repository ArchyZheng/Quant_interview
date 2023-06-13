# this file is used for dataset module related lightning, which can generator train loader and validation loader
# %%
import lightning
from torch.utils.data import DataLoader
import torch

TRAIN_DATALOADERS = DataLoader
EVAL_DATALOADERS = DataLoader


class DatasetModule(lightning.LightningDataModule):
    def __init__(self, dataset, batch_size: int):
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.dataset = dataset
        self.batch_size = batch_size

    def setup(self, stage='None') -> None:
        train_dataset_length = int(len(self.dataset) * 0.8)
        val_dataset_length = int(len(self.dataset) * 0.2)
        self.train_dataset = torch.utils.data.Subset(self.dataset, range(train_dataset_length))
        self.val_dataset = torch.utils.data.Subset(self.dataset, range(train_dataset_length,
                                                                       train_dataset_length + val_dataset_length))

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(dataset=self.val_dataset, batch_size=100, shuffle=True, drop_last=True)
