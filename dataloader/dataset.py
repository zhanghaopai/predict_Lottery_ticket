import argparse

import pandas as pd
from torch.utils.data import Dataset
from loguru import logger


class SSQDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.path = args.path
        self.data = pd.read_csv(self.path, delimiter=',', header='infer')
        # 分割训练集和测试集
        train_size = int(len(self.data) * args.train_ratio)
        self.train_set = self.data[:train_size]
        self.test_set = self.data[train_size:]
        logger.info("读取到【训练集】长度为：{}", len(self.train_set))
        logger.info("读取到【测试集】长度为：{}", len(self.test_set))


    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.path='../data/ssq/data.csv'
    args.train_ratio=0.8
    SSQDataset(args)
