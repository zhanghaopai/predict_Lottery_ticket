import argparse

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from loguru import logger


class SSQDataset(Dataset):
    def __init__(self, args, train=True):
        self.args = args
        self.path = args.path
        self.data = pd.read_csv(self.path, delimiter=',', skiprows=1,
                                names=['index', 'time', 'red_1', 'red_2', 'red_3', 'red_4', 'red_5', 'red_6', 'blue'])
        # 分割训练集和测试集
        train_size = int(len(self.data) * args.train_ratio)
        if train:
            self.set = self.data[:train_size]
        else:
            self.set = self.data[train_size:]

    def __len__(self):
        return len(self.set)

    def __getitem__(self, idx):
        return self.set.iloc[idx]


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.path='../data/ssq/data.csv'
    args.train_ratio=0.8
    train_set = SSQDataset(args, train=True)
    test_set = SSQDataset(args, train=False)
    logger.info("train_set length:{}", len(train_set))
    logger.info("test_set length:{}", len(test_set))
