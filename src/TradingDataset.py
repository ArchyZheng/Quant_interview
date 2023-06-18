# this file is used to build new dataset for interview of quant finance
# %%
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
from utility.MeanAndStdofAll import get_mean_and_std


class TradingDataset(Dataset):
    def __init__(self, data_source: str, data_filename: str,
                 metadata=None):
        """

        :param data_source: The folder of data
        :param data_filename: the full name of data file
        :param metadata: the first element should be 'date' or 'datetime' as the index of dataframe.
        """
        super().__init__()
        if metadata is None:
            metadata = ['datetime', 'open', 'close', 'high', 'close', 'volume']
        assert metadata[1] == 'open' and metadata[2] == 'close'  # Safe from bug
        self.file = os.path.join(data_source, data_filename)
        self.metadata = metadata
        self.data = pd.read_csv(self.file, usecols=self.metadata, index_col=self.metadata[0])
        # get the mean and std value from ['data_IC_15m.csv', 'data_IF_15m.csv', 'data_IH_15m.csv', 'data_IM_15m.csv']
        data_name = ['data_IC_15m.csv', 'data_IF_15m.csv', 'data_IH_15m.csv', 'data_IM_15m.csv']
        mean_df, std_df = get_mean_and_std(data_source=data_source, data_filename_list=data_name)
        # adjust the order of columns
        self.mean = mean_df[self.metadata[1:]]
        self.std = std_df[self.metadata[1:]]
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
        x = x / self.mean[left:mid].values
        x = (x - self.mean[left:mid].values) / self.std[left:mid].values

        # the index of open: 1, the index of close: 2
        # the open price of the midday - the close price of the end day.
        y = (self.X[right][2] - self.X[mid][1]) / self.X[mid][1]
        return x, y

    def __len__(self):
        return len(self.data) - 30 * 16
