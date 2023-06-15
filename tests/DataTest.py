import datetime
import pprint
import unittest

import torch

from src.TradingDataset import TradingDataset
from src.DatasetModule import DatasetModule


class DataTests(unittest.TestCase):
    def test_Dataset_creation(self):
        """

        Test 000001sz dataset:

        the length of dataset:
        1. assertEqual 38944 - 30 * 16
        """
        data_source = '/Users/archy/PycharmProjects/trial_quant/data'
        data_filename = '000001sz.csv'
        metadata = ['time', 'open', 'close', 'high', 'low', 'volume']

        dataset = TradingDataset(data_source=data_source, data_filename=data_filename, metadata=metadata)
        self.assertEqual(len(dataset), 38944 - 30 * 16)


    def test_DatasetModule(self):
        """
        test dataloader:
        1. train_dataloader
        2. val_dataloader

        test the order of dataset:
        1. print the first ten data series
        """
        data_source = '/Users/archy/PycharmProjects/trial_quant/data'
        data_filename = '000001sz.csv'
        metadata = ['time', 'open', 'close', 'high', 'low', 'volume']
        dataset = TradingDataset(data_source=data_source, data_filename=data_filename, metadata=metadata)

        datamodule = DatasetModule(dataset=dataset, batch_size=10)
        datamodule.setup()  # this is important for datamodule!
        train_dataloader = datamodule.train_dataloader()
        val_dataloader = datamodule.val_dataloader()
        pprint.pp("Train_dataloader:")
        pprint.pp(next(iter(train_dataloader)))
        pprint.pp("Validation_dataloader:")
        pprint.pp(next(iter(val_dataloader)))

if __name__ == '__main__':
    unittest.main()
