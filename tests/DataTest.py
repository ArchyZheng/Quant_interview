import pprint
import unittest
from src.Dataset import Dataset
import os


class DataTests(unittest.TestCase):
    def test_Dataset_creation(self):
        """
        Using different data_name:
        1. data_IC_15m
        2. data_IF_15m

        Test the length of dataset and output.
        """
        data_names = ['data_IC_15m', 'data_IF_15m']
        data_length = [24928, 24928]
        dataset_list = []
        data_source = '/Users/archy/PycharmProjects/trial_quant/data'
        for name in data_names:
            dataset = Dataset(data_source=data_source, data_name=name)
            dataset_list.append(dataset)

        for index in range(2):
            self.assertEqual(len(dataset_list[index]), data_length[index])
            pprint.pp(dataset_list[index].data.head(5))


if __name__ == '__main__':
    unittest.main()
