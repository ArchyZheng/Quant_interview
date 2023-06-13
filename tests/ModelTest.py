import unittest

import torch

from src.TradingDataset import TradingDataset
from src.DatasetModule import DatasetModule
from src.Model import BaseGRUModel


class ModelTest(unittest.TestCase):
    def test_prediction_shape(self):
        """
        if using the batch size = 5000
        the shape of each tensor:
        1. target: [5000]
        2. prediction: [5000]
        3. input: [5000, 320, 5]
        """
        data_source = '/Users/archy/PycharmProjects/trial_quant/data'
        dataset = TradingDataset(data_source=data_source, data_name='data_IC_15m')
        dataset_module = DatasetModule(dataset=dataset, batch_size=5000)
        dataset_module.setup()
        train_dataloader = dataset_module.train_dataloader()

        model = BaseGRUModel(input_size=6, hidden_size=30, num_layers=1, with_attention=False)
        h_0 = model.init_hidden(batch_size=5000)
        for input_feature, target in train_dataloader:
            prediction, _ = model(input_feature, h_0)
            self.assertEqual(prediction.shape, torch.Size([5000]))
            self.assertEqual(input_feature.shape, torch.Size([5000, 320, 6]))
            self.assertEqual(target.shape, torch.Size([5000]))


if __name__ == '__main__':
    unittest.main()
