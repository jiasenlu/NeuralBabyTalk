import os

import h5py
from torch.utils.data import Dataset


class HDFShardDataset(Dataset):
    def __init__(self, shard_dir, shard_names=None, primary_key=None, stride=1):
        super().__init__()
        self.shard_dir = shard_dir
        self.shard_names = shard_names
        if not shard_names:
            self.shard_names = sorted(os.listdir(shard_dir))
        self.primary_key = self.__primary_key(primary_key)
        self.stride = stride

        # length is expressed as per items, not rows (#items * stride = #rows)
        self.shard_len, self.dataset_len = self.__shard_len_dataset_len()

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        shard_num = idx // self.shard_len
        idx -= shard_num * self.shard_len
        nth_shard = h5py.File(os.path.join(self.shard_dir, self.shard_names[shard_num]), 'r')
        keys = list(nth_shard.keys())
        item = {}
        for key in keys:
            item[key] = nth_shard[key][idx * self.stride : (idx + 1) * self.stride]
        nth_shard.close()
        return item

    def __primary_key(self, primary_key):
        first_shard = h5py.File(os.path.join(self.shard_dir, self.shard_names[0]), 'r')
        if not primary_key:
            primary_key = list(first_shard.keys())[0]
        first_shard.close()
        return primary_key

    def __shard_len_dataset_len(self):
        # check number of items per shard by opening one shard
        # check remainder number of items in last shard
        first_shard = h5py.File(os.path.join(self.shard_dir, self.shard_names[0]), 'r')
        last_shard = h5py.File(os.path.join(self.shard_dir, self.shard_names[-1]), 'r')
        rows_per_shard = len(first_shard[self.primary_key])
        rows_per_last_shard = len(last_shard[self.primary_key])

        dataset_len = rows_per_shard * (len(self.shard_names) - 1) // self.stride
        dataset_len += rows_per_last_shard // self.stride
        shard_len = rows_per_shard // self.stride
        first_shard.close()
        last_shard.close()
        return shard_len, dataset_len


class HDFSingleDataset(HDFShardDataset):
    def __init__(self, hdf_path, primary_key=None, stride=1):
        super().__init__(
            os.path.dirname(hdf_path),
            shard_names=[os.path.basename(hdf_path)],
            primary_key=primary_key,
            stride=stride,
        )
