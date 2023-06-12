# this file is used to build new dataset for interview of quant finance
# %%
from torch.utils.data import Dataset
import pandas as pd
import os

class Dataset(Dataset):
    def __init__(self, data_source: str, data_name: str):
        """

        :param data_source: the path of data
        :param data_name:  the name of data, enum: {data_IC_15m, data_IF_15m, data_IH_15m, data_IM_15m}
        """
        super().__init__()
        data_path = os.path.join(data_source, data_name + '.csv')
        self.data = pd.read_csv(data_path, header=0)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
# %%
os.getcwd()