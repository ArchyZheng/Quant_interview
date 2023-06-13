import unittest
import torch
from src.TrainModule import ic_loss

class LossFunctionTest(unittest.TestCase):
    def test_loss(self):
        """
        test one example:
        X = [0, 1, 2]
        Y = [2, 1, 0]
        EX = 1
        EY = 1
        X - EX = [-1, 0, 1]
        Y - EY = [1, 0, -1]
        (X - EX)(Y - EY) = [-1, 0, -1]
        cov(X, Y) = -2/3
        DX = EX^2 - (EX)^2 = 5/3 - 1 = 2/3
        DY = 5/3 - 1 = 2/3

        rho_xy = -2/3 / 2/3 = -1
        the loss is the invert of rho_xy = 1
        """
        prediction = torch.tensor([0, 1, 2])
        target = torch.tensor([2, 1, 0])
        loss = ic_loss(prediction, target)
        self.assertEqual(loss, torch.tensor(1))

if __name__ == '__main__':
    unittest.main()
