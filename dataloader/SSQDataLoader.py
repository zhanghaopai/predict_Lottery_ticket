from torch.utils.data import DataLoader
from dataset import SSQDataset

def build_dataloader(args):
    train_set=SSQDataset(args, train=True)
    test_set=SSQDataset(args, train=False)
    train_dataloader=DataLoader(train_set)
    test_dataloader=DataLoader(test_set,
                               batch_size=args.batch_size,
                               shuffle=True)
    return train_dataloader, test_dataloader
