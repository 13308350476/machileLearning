import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class DiabetesDatasets(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, -1:])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

print('!!!!!!!!!!')
class LinearModle(torch.nn.Module):
    def __init__(self):
        super(LinearModle, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.active = torch.nn.Softsign()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.active(self.linear1(x))
        x = self.active(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

if __name__ == '__main__':

    datasets = DiabetesDatasets('../datasets/diabetes/diabetes.csv.gz')
    train_loader = DataLoader(dataset=datasets, batch_size=32, shuffle=True, num_workers=2) 

    modle = LinearModle()

    criteration = torch.nn.BCELoss(reduction='mean')
    optimition = torch.optim.SGD(modle.parameters(), lr=0.01)
    print('!!!!!!!!!!')
    for epoch in range(10000):
        print("epoch:", epoch)
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            y_prey = modle(inputs)
            loss = criteration(y_prey, labels)
            if(i == 0):
                print(loss.item())
            optimition.zero_grad()
            loss.backward()
            optimition.step()

    xy = np.loadtxt('../datasets/diabetes/diabetes.csv.gz', delimiter=',', dtype=np.float32)
    x_test = torch.from_numpy(xy[-5:, :-1])
    y_test = modle(x_test)

    print('y_test:', y_test) 