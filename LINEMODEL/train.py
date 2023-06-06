import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class DiateatsDatasets(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimter=',', dtype=np.float32)
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])
        self.len = xy.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len

class LinearModule(torch.nn.Module):
    def __init__(self):
        super(LinearModule, self).__init__()
        self.linear = torch.nn.Linear(8, 1)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        y_prey = self.sigmoid(self.linear(x))
        return y_prey
    
if __name__ == '__main__':
    datasets = DiateatsDatasets('/datasets/diabetes/diabetes.csv.gz')
    

