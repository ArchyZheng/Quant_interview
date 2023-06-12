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
        col_name_type = {'volume': torch.float16, 'high': torch.float16, 'close': torch.float16, 'low': torch.float16,
                'open': torch.float16}
        self.data = pd.read_csv(data_path, header=0, usecols=col_name_type.keys())
        self.x = self.data.iloc[:, 0:5].values
        self.x = torch.tensor(self.x, dtype=torch.float32)

    def __getitem__(self, item):
        """

        :param item: the index of data
        :return: the volume, high, close, low, open depending on index
        """
        return self.x[item]

    def __len__(self):
        """

        :return: the length of dataset
        """
        return len(self.x)
