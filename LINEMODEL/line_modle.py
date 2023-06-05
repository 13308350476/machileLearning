import torch

x_data = torch.tensor([[1.0], [2.0], [3.0]]) #line sampel  row feature
y_data = torch.tensor([[2.0], [4.0], [6.0]])

print(x_data.shape)

class LinearModel(torch.nn.Module):
    def __init__(self) :
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = LinearModel()

criteration = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1):
    y_pred = model(x_data)
    loss = criteration(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('w=', model.linear.weight.item())
print('b=', model.linear.bias.item())

x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print('y_pred=', y_test.data)