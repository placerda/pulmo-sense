import unittest
import os
from dotenv import load_dotenv
import torch
from datasets.mosmed_dataset import MosMedDataset

class TestMosMedDataset(unittest.TestCase):
    def setUp(self):
        load_dotenv()
        self.root_dir = os.getenv('PATH_TO_MOSMED_DATASET')
        self.dataset = MosMedDataset(self.root_dir)

    def test_len(self):
        expected_len = len(os.listdir(os.path.join(self.root_dir, 'images')))
        self.assertEqual(len(self.dataset), expected_len)

    def test_getitem(self):
        item = self.dataset[0]
        self.assertIsInstance(item, tuple)
        self.assertIsInstance(item[0], torch.Tensor)
        self.assertIsInstance(item[1], torch.Tensor)

if __name__ == '__main__':
    unittest.main()