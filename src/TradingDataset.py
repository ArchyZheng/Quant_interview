# this file is used to build new dataset for interview of quant finance
# %%
from torch.utils.data import Dataset
import pandas as pd
import os
import torch


class TradingDataset(Dataset):
    def __init__(self, data_source: str, data_name: str):
        """

        :param data_source: the path of data
        :param data_name:  the name of data, enum: {data_IC_15m, data_IF_15m, data_IH_15m, data_IM_15m}
        """
        super().__init__()
        data_path = os.path.join(data_source, data_name + '.csv')
        col_name_type = {'volume': torch.float32, 'high': torch.float32, 'close': torch.float32, 'low': torch.float32,
                         'open': torch.float32}
        self.data = pd.read_csv(data_path, header=0, usecols=col_name_type.keys())
        # Normalization for feature
        for col_name in col_name_type.keys():
            self.data[col_name] = self.data[col_name] / self.data[col_name].mean()
            self.data[col_name] = (self.data[col_name] - self.data[col_name].mean()) / self.data[col_name].std()

        # TODO: Normalization for target, get the target mean and std

        self.data = self.data.iloc[:, 0:5].values
        self.data = torch.tensor(self.data, dtype=torch.float32)

    def __getitem__(self, item):
        """

        :param item: the index of middle data
        :return: the volume, high, close, low, open depending on index [mid - 20: mid], the output of yield rate
        """
        left = item
        mid = item + 20 * 16
        right = item + 30 * 16
        x = self.data[left:mid].reshape(-1, 5)
        # the index of close: 2, the index of open: 4
        y = (self.data[mid:right][:, 2] - self.data[mid:right][:, 4]) / self.data[mid:right][:, 4]
        return x, torch.sum(y)

    def __len__(self):
        """

        :return: the length of dataset
        """
        return len(self.data) - 30 * 16
