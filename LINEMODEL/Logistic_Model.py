import torch
import torch.nn.functional as F

x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = torch.tensor([[0.0], [0.0], [1.0], [1.0]])


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
    
    def forward(self,x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred
    
model = LogisticRegressionModel()

criteration = torch.nn.BCELoss(reduction='sum')
optimition = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(2000):
    y_prey = model(x_data)
    loss = criteration(y_prey, y_data)
    print(epoch, loss.item())

    optimition.zero_grad()
    loss.backward()
    optimition.step()

print('w=', model.linear.weight.item())
print('b=', model.linear.bias.item())

x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print('y_pred=', y_test.data)