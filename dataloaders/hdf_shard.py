import glob
import os

import h5py

from torch.utils.data import Dataset


class HDFShardDataset(Dataset):
    def __init__(self, shard_dir, primary_key=None, stride=100):
        self.shard_dir = shard_dir
        self.shard_names = sorted(os.listdir(shard_dir))
        self.primary_key = self.__primary_key(primary_key)
        self.stride = stride
        self.dataset_len = self.__dataset_len()

    def __len__(self):
        return self.dataset_len

    def __primary_key(self, primary_key):
        first_shard = h5py.File(
            os.path.join(self.shard_dir, self.shard_names[0]))
        if not primary_key:
            primary_key = list(first_shard.keys())[0]
        first_shard.close()
        return primary_key

    def __dataset_len(self):
        # check number of items per shard by opening one shard
        # check remainder number of items in last shard
        first_shard = h5py.File(
            os.path.join(self.shard_dir, self.shard_names[0]))
        last_shard = h5py.File(
            os.path.join(self.shard_dir, self.shard_names[-1])
        )
        rows_per_shard = len(first_shard[self.primary_key])
        rows_per_last_shard = len(last_shard[self.primary_key])

        dataset_len = (rows_per_shard * (len(self.shard_names) - 1) // self.stride) 
        dataset_len += rows_per_last_shard // self.stride
        first_shard.close()
        last_shard.close()
        return dataset_len
