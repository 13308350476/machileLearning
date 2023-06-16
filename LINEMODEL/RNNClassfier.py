import torch
from torch.utils.data import Dataset

class DataSet(Dataset):
    def __init__(self, is_trainset=True):
        file_name = '../datasets/name/names_train.csv.gz' if is_trainset else '../datasets/name/names_test.csv.gz'
        