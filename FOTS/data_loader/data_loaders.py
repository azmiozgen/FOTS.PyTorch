import os

from lib2to3.pytree import Base
import torch.utils.data as torchdata

from ..base import BaseDataLoader
from .dataset import PriceTagDataset, SynthTextDataset
from .datautils import collate_fn


class PriceTagDataLoaderFactory(BaseDataLoader):

    def __init__(self, config):
        super(PriceTagDataLoaderFactory, self).__init__(config)
        data_root = self.config['data_loader']['data_dir']
        train_data_root = os.path.join(data_root, 'train')
        val_data_root = os.path.join(data_root, 'val')
        self.train_dataset = PriceTagDataset(train_data_root)
        self.val_dataset = PriceTagDataset(val_data_root)
        self.workers = self.config['data_loader']['workers']

    def train(self):
        # train_loader = torchdata.DataLoader(self.train_dataset, num_workers=self.num_workers, batch_size=self.batch_size,
        #                                    shuffle=self.shuffle, collate_fn=collate_fn)
        train_loader = torchdata.DataLoader(self.train_dataset, num_workers=self.num_workers, batch_size=self.batch_size,
                                           shuffle=self.shuffle)
        return train_loader

    def val(self):
        shuffle = self.config['validation']['shuffle']
        # val_loader = torchdata.DataLoader(self.val_dataset, num_workers=self.num_workers, batch_size=self.batch_size,
        #                                  shuffle=shuffle, collate_fn=collate_fn)
        val_loader = torchdata.DataLoader(self.val_dataset, num_workers=self.num_workers, batch_size=self.batch_size,
                                         shuffle=shuffle)
        return val_loader

class SynthTextDataLoaderFactory(BaseDataLoader):

    def __init__(self, config):
        super(SynthTextDataLoaderFactory, self).__init__(config)
        dataRoot = self.config['data_loader']['data_dir']
        self.workers = self.config['data_loader']['workers']
        ds = SynthTextDataset(dataRoot)
        
        self.__trainDataset, self.__valDataset = self.__train_val_split(ds)

    def train(self):
        trainLoader = torchdata.DataLoader(self.__trainDataset, num_workers = self.num_workers, batch_size = self.batch_size,
                                           shuffle = self.shuffle, collate_fn = collate_fn)
        return trainLoader

    def val(self):
        if self.__valDataset is None:
            return None

        shuffle = self.config['validation']['shuffle']
        valLoader = torchdata.DataLoader(self.__valDataset, num_workers = self.num_workers, batch_size = self.batch_size,
                                         shuffle = shuffle, collate_fn = collate_fn)
        return valLoader

    def __train_val_split(self, ds):
        '''

        :param ds: dataset
        :return:
        '''
        split = self.config['validation']['validation_split']

        try:
            split = float(split)
        except:
            raise RuntimeError('Train and val splitting ratio is invalid.')

        if not split:
            return ds, None


        val_len = int(split * len(ds))
        train_len = len(ds) - val_len
        train, val = torchdata.random_split(ds, [train_len, val_len])
        return train, val

    def split_validation(self):
        raise NotImplementedError


class OCRDataLoaderFactory(BaseDataLoader):

    def __init__(self, config, ds):
        super(OCRDataLoaderFactory, self).__init__(config)
        self.workers = self.config['data_loader']['workers']
        self.__trainDataset, self.__valDataset = self.__train_val_split(ds)

    def train(self):
        trainLoader = torchdata.DataLoader(self.__trainDataset, num_workers = self.num_workers, batch_size = self.batch_size,
                                           shuffle = self.shuffle, collate_fn = collate_fn)
        return trainLoader

    def val(self):
        if self.__valDataset is None:
            return None

        shuffle = self.config['validation']['shuffle']
        valLoader = torchdata.DataLoader(self.__valDataset, num_workers = self.num_workers, batch_size = self.batch_size,
                                         shuffle = shuffle, collate_fn = collate_fn)
        return valLoader

    def __train_val_split(self, ds):
        '''

        :param ds: dataset
        :return:
        '''
        split = self.config['validation']['validation_split']

        try:
            split = float(split)
        except:
            raise RuntimeError('Train and val splitting ratio is invalid.')

        if not split:
            return ds, None

        val_len = int(split * len(ds))
        train_len = len(ds) - val_len
        train, val = torchdata.random_split(ds, [train_len, val_len])
        return train, val

    def split_validation(self):
        raise NotImplementedError