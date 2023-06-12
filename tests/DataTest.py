import pprint
import unittest
from src.TradingDataset import TradingDataset
from src.DatasetModule import DatasetModule
import os


class DataTests(unittest.TestCase):
    def test_Dataset_creation(self):
        """
        Using different data_name:
        1. data_IC_15m
        2. data_IF_15m

        Test the length of dataset and output:
        """
        data_names = ['data_IC_15m', 'data_IF_15m']
        data_length = [24928 - 30, 24928 - 30]
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

        for dataset in dataset_list:
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
