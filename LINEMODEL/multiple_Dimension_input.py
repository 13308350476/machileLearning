import torch
import numpy as np

xy = np.loadtxt('/datasets/diabetes/diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_date = torch.from_numpy(xy[:-5, :-1])
y_date = torch.from_numpy(xy[:-5, -1:])

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
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
    
model = LinearModel()

criteration = torch.nn.BCELoss(reduction='sum')
optimition = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100000):
    y_prey = model(x_date)
    loss = criteration(y_prey, y_date)
    print(epoch, loss.item())

    optimition.zero_grad()
    loss.backward()
    optimition.step()

x_test = torch.from_numpy(xy[-5:, :-1])
y_test = model(x_test)

print('y_test:', y_test) 

