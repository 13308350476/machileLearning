import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class DiateatsDatasets(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
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
    datasets = DiateatsDatasets('../datasets/diabetes/diabetes.csv.gz')
    train_loader  = DataLoader(dataset=datasets, shuffle=True, batch_size=32, num_workers=0)

    module = LinearModule()
    criteration = torch.nn.BCELoss(reduction='mean')
    optimition = torch.optim.SGD(module.parameters(), lr=0.1)

    for epoch in range(100):
        for i, data in enumerate(train_loader, 0):
            x_data, y_data = data
            y_prey = module(x_data)
            loss = criteration(y_prey, y_data)
            print(epoch, i, loss.item())
            optimition.zero_grad()
            loss.backward()
            optimition.step()
    
        xy = np.loadtxt('../datasets/diabetes/diabetes.csv.gz', delimiter=',', dtype=np.float32)
        x_test = torch.from_numpy(xy[-5:, :-1])
        y_test = module(x_test)

        print('y_test:', y_test) 

