import unittest
import torch
from utility.MeanAndStdofAll import get_mean_and_std


class MeanAndStdTest(unittest.TestCase):
    def test_mean_and_std(self):
        """

        we should read four data file, and get the mean and std for each column:
        1. print the dataframe for each data, test the first index and second index. Had been tested, in the utility!
        2. get the first mean and std of the volume.
        """
        mean_volume = torch.mean(torch.tensor([820.0, 1368.0, 673.0]))
        std_volume = torch.std(torch.tensor([820.0, 1368.0, 673.0]))

        data_source = '/Users/archy/PycharmProjects/quant_interview/data'
        data_name = ['data_IC_15m.csv', 'data_IF_15m.csv', 'data_IH_15m.csv', 'data_IM_15m.csv']
        mean_df, std_df = get_mean_and_std(data_source=data_source, data_filename_list=data_name)
        self.assertEqual(torch.tensor(mean_df['volume'].iloc[0], dtype=torch.float32), mean_volume)
        self.assertEqual(torch.tensor(std_df['volume'].iloc[0], dtype=torch.float32), std_volume)


if __name__ == '__main__':
    unittest.main()
