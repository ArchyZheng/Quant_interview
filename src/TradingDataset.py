# this file is used to build new dataset for interview of quant finance
# %%
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import datetime


class TradingDataset(Dataset):
    def __init__(self, data_source: str, data_filename: str,
                 metadata: list = ['datetime', 'open', 'close', 'high', 'close','volume']):
        """

        :param data_source: The folder of data
        :param data_filename: the full name of data file
        :param metadata: the first element should be 'date' or 'datetime' as the index of dataframe.
        """
        super().__init__()
        assert metadata[1] == 'open' and metadata[2] == 'close' # Safe from bug
        self.file = os.path.join(data_source, data_filename)
        self.metadata = metadata
        self.data = pd.read_csv(self.file, usecols=self.metadata, index_col=self.metadata[0])
        self.data = self.data[self.metadata[1:]]  # adjust the order of columns. Ready for change.
        self.data.sort_index(inplace=True)  # Safe from bugs

        self.X = self.data.iloc[:, :].values
        self.X = torch.tensor(self.X)

    def __getitem__(self, item):
        """

        :param item: not negative
        """
        left = item
        mid = item + 20 * 16
        right = item + 30 * 16
        # normalization
        x = self.X[left:mid]
        x = x / x.mean(dim=0)
        x = (x - x.mean(dim=0)) / x.std(dim=0)

        # the index of open: 1, the index of close: 2
        y = (self.X[mid:right][:, 2] - self.X[mid:right][:, 1]) / self.X[mid:right][:, 1]
        y = (y - y.mean()) / y.std()
        return x, torch.sum(y)

    def __len__(self):
        return len(self.data) - 30 * 16