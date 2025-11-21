import torch
import torch.nn
import scipy.io as sio
import numpy as np
from torch.utils.data import DataLoader
# from Sampler import MetaSampler
# from dataset import MyDataset
from dataset.dataset import CUB
import torchnet as tnt
import copy
from collections import defaultdict
import random
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


class MetaDataLoader(object):
    def __init__(self, data, labels, class_per_it=5, num_shot=5, iter_num=1000, num_workers=4):
        super(MetaDataLoader, self).__init__()
        self.data = data
        self.label = labels
        self.num_workers = num_workers
        self.length = len(labels)
        self.iter_num = iter_num
        self.labels = copy.deepcopy(labels.numpy())
        self.label2index = defaultdict(list)
        for index, label in enumerate(self.labels):
            self.label2index[label].append(index)
        self.all_labels = sorted(list(self.label2index.keys()))
        print(self.all_labels)
        self.num_classes = len(self.all_labels)
        self.sample_per_class = num_shot
        self.classes_per_it = class_per_it

    def get_train_task(self, idx):
        # yield a batch of indexes
        spc = self.sample_per_class
        cpi = self.classes_per_it

        # batch_size = spc * cpi
        few_shot_train_batch = []
        few_shot_valid_batch = []
        if cpi <= len(self.all_labels):
            batch_few_shot_classes = random.sample(self.all_labels, cpi)
        else:
            batch_few_shot_classes = np.random.choice(self.all_labels, cpi, True).tolist()
        for i, cls in enumerate(batch_few_shot_classes):
            if len(self.label2index[cls]) < spc:
                continue
            sample_indexes = random.sample(self.label2index[cls], spc)
            few_shot_train_batch.extend(sample_indexes)
        return few_shot_train_batch

    def get_test_task(self, index):
        return self.data[index], self.label[index]

    def get_iterator(self):
        sampler = self.get_train_task
        tnt_dataset = tnt.dataset.ListDataset(
            elem_list=range(self.iter_num), load=sampler)
        data_loader = tnt_dataset.parallel(
            batch_size=1,
            # num_workers=(1 if self.is_eval_mode else self.num_workers),
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )
        return data_loader

    def __call__(self):
        return self.get_iterator()


if __name__ == "__main__":
    data = CUB()
    DL = MetaDataLoader(
        data=data.train_x,

    )