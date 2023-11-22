from unittest import TestCase
from torch import Tensor, LongTensor
from trainer.loss import accuracy, Loss, LossArgs


class LossAndAccuracyTest(TestCase):
    def setUp(self):
        self.output = Tensor([
            [[0.2, 0.5, 0.9, 0.1], [0.2, 0.5, 0.9, 0.1]],
            [[0.2, 0.5, 0.4, 7], [0.042, 0.05, 0.2, 0.1]],
        ])
        self.target = LongTensor([
            [1, 2],  # NO OK, OK
            [3, 2]  # OK, OK
        ])
        self.accuracy = 0.75  # 3/4
        self.loss_module = Loss(LossArgs())

    def test_accuracy(self):
        # corpus is created
        self.assertEqual(accuracy(self.output, self.target), self.accuracy)
