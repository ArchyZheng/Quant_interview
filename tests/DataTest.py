import pprint
import unittest

import torch

from src.TradingDataset import TradingDataset
from src.DatasetModule import DatasetModule


class DataTests(unittest.TestCase):
    def test_Dataset_creation(self):
        """
        Using different data_name:
        1. data_IC_15m
        2. data_IF_15m

        Test the length of dataset and output:
        """
        data_names = ['data_IC_15m', 'data_IF_15m']
        data_length = [24928 - 30 * 16, 24928 - 30 * 16]
        dataset_list = []
        data_source = '/Users/archy/PycharmProjects/trial_quant/data'
        for name in data_names:
            dataset = TradingDataset(data_source=data_source, data_name=name)
            dataset_list.append(dataset)

        for index in range(2):
            self.assertEqual(len(dataset_list[index]), data_length[index])
            pprint.pp(dataset_list[index].data[:5])

    def test_DatasetModule(self):
        """
        test dataloader:
        1. train_dataloader
        2. val_dataloader

        test the order of dataset:
        1. print the first ten data series
        """
        data_source = '/Users/archy/PycharmProjects/trial_quant/data'
        name_list = ['data_IC_15m', 'data_IF_15m']
        dataset_list = []
        for name in name_list:
            dataset = TradingDataset(data_source=data_source, data_name=name)
            dataset_list.append(dataset)

        for dataset in dataset_list:  # if test the dataloader, please switch the `shuffle` of dataloader.
            datamodule = DatasetModule(dataset=dataset, batch_size=10)
            datamodule.setup()  # this is important for datamodule!
            train_dataloader = datamodule.train_dataloader()
            val_dataloader = datamodule.val_dataloader()
            pprint.pp("Train_dataloader:")
            pprint.pp(next(iter(train_dataloader)))
            pprint.pp("Validation_dataloader:")
            pprint.pp(next(iter(val_dataloader)))

    def test_vwap(self):
        """
        we will test IC data, for the first five datapoint:

        vwap = total_turnover / volume/ lot size
        vwap = total_turnover / volume / 200

        total_turnover: 1023269520.0, 1232513600.0, 634842840.0, 659663960.0, 568840200.0
        volume: 820.0, 983.0, 506.0, 525.0, 452.0

        """
        total_turnover = torch.tensor([1023269520.0, 1232513600.0, 634842840.0, 659663960.0, 568840200.0])
        volume = torch.tensor([820.0, 983.0, 506.0, 525.0, 452.0])
        vwap = total_turnover / volume / 200.0

        data_source = '/Users/archy/PycharmProjects/trial_quant/data'
        data_name = 'data_IC_15m'
        dataset = TradingDataset(data_source=data_source, data_name=data_name)
        for i in range(5):
            self.assertEqual(dataset.for_testing[i], vwap[i])
            # when index is 2, this test will fail. I think it may cause by precision of python


if __name__ == '__main__':
    unittest.main()
