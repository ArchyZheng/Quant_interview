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
        lot_size = {'data_IC_15m': 200, 'data_IF_15m': 300, 'data_IH_15m': 300, 'data_IM_15m': 200}
        col_name_type = {'volume': torch.float32, 'high': torch.float32, 'close': torch.float32, 'low': torch.float32,
                         'total_turnover': torch.float32, 'open': torch.float32, 'dominant_id': str}
        self.data = pd.read_csv(data_path, header=0, usecols=col_name_type.keys())
        # Create VWAP
        self.data['total_turnover'] = self.data['total_turnover'] / self.data['volume'] / lot_size[data_name]
        # This part is created for testing vwap
        self.for_testing = self.data['total_turnover'].head(5)
        self.for_testing = self.for_testing[:].values
        self.for_testing = torch.tensor(self.for_testing, dtype=torch.float32)

        self.data.rename(columns={'total_turnover': 'vwap'}, inplace=True)
        self.data = self.data.iloc[:, 0:6].values
        self.data = torch.tensor(self.data, dtype=torch.float32)

    def __getitem__(self, item):
        """

        :param item: the index of middle data
        :return: the volume, high, close, low, open depending on index [mid - 20: mid], the output of yield rate
        """
        left = item
        mid = item + 20 * 16
        right = item + 30 * 16
        x = self.data[left:mid].reshape(-1, 6)
        x = x / x.mean(dim=0)
        x = (x - x.mean(dim=0)) / x.std(dim=0)  # had been tested!
        # the index of close: 2, the index of open: 5
        y = (self.data[mid:right][:, 2] - self.data[mid:right][:, 5]) / self.data[mid:right][:, 5]
        # y = (y - y.mean()) / y.std()  # 因为它是一个因子, 所以不必和真实的收益率直接挂钩
        return x, torch.sum(y)

    def __len__(self):
        """

        :return: the length of dataset
        """
        return len(self.data) - 30 * 16
