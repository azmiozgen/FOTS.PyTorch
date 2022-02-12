import os

import torch.utils.data as torchdata

from ..base import BaseDataLoader
from .dataset import PriceTagDataset
from .datautils import collate_fn


class PriceTagDataLoaderFactory(BaseDataLoader):

    def __init__(self, config):
        super(PriceTagDataLoaderFactory, self).__init__(config)
        base_dir = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir))
        datasets_dir = os.path.join(base_dir, 'datasets')
        data_dirname = self.config['data_loader']['data_dirname']
        data_root = os.path.join(datasets_dir, data_dirname)
        train_data_root = os.path.join(data_root, 'train')
        val_data_root = os.path.join(data_root, 'val')
        test_data_root = os.path.join(data_root, 'test_small_small')
        self.train_dataset = PriceTagDataset(train_data_root, config=config, train_mode=True)
        self.val_dataset = PriceTagDataset(val_data_root, config=config, train_mode=False)
        self.test_dataset = PriceTagDataset(test_data_root, config=config, train_mode=False)
        self.workers = self.config['data_loader']['workers']

    def train(self):
        train_loader = torchdata.DataLoader(self.train_dataset,
                num_workers=self.num_workers,
                batch_size=min(len(self.train_dataset), self.batch_size),
                shuffle=self.shuffle,
                collate_fn=collate_fn)
        return train_loader

    def val(self):
        shuffle = self.config['validation']['shuffle']
        val_loader = torchdata.DataLoader(self.val_dataset,
                num_workers=self.num_workers,
                batch_size=min(len(self.val_dataset), self.batch_size),
                shuffle=shuffle,
                collate_fn=collate_fn)
        return val_loader

    def test(self):
        batch_size = self.config['tester']['batch_size']
        test_loader = torchdata.DataLoader(self.test_dataset,
                num_workers=self.num_workers,
                batch_size=min(len(self.test_dataset), batch_size),
                shuffle=True,
                collate_fn=collate_fn)
        return test_loader
