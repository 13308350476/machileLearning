import torch
from torch.utils.data import Dataset
import gzip
import csv

class NameDataSet(Dataset):
    def __init__(self, is_trainset=True):
        file_name = '../datasets/name/names_train.csv.gz' if is_trainset else '../datasets/name/names_test.csv.gz'
        with gzip.open(file_name, 'rt') as f:
            reader = csv.reader(f)
            rows = list(reader)
        self.names = [rows[0] for row in rows]
        self.len = len(self.names)
        self.contries = [rows[1] for row in rows]
        self.country_list = list(sorted(set(self.contries)))
        self.country_dict = self.getCountryDict()
        self.contry_num = len(self.country_list)
    
    def __getitem__(self, index):
        return self.names[index], self.country_dict[self.contries[index]]
    
    def __len__(self):
        return self.len
    
    def getCountryDict(self):
        conunry_dict = dict()
        for idx, countryname in enumerate(self.country_list, 0):
            conunry_dict[countryname] = idx
        return conunry_dict
    
    def id2xcountry(self, index):
        return self.country_list(index)
    
    def getCountriesNum(self):
        return self.contry_num


        
    
class NET(torch.nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        