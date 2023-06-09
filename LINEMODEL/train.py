import torch 
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
batch_size = 64

train_set = datasets.MNIST(root='../datasets/minist', train=True, download=True, transform=transform)
train_load = DataLoader(dataset=train_set, shuffle=True, batch_size=batch_size)

test_set = datasets.MNIST(root='../datasets/minist', train=False, download=True, transform=transform)
test_load = DataLoader(dataset=test_set, shuffle=False, batch_size=batch_size)

class NET(torch.nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        self.l1 = torch.nn.Linear(784, 128)
        self.l2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x
    
modle = NET()

criteration = torch.nn.CrossEntropyLoss()
optimition = torch.optim.SGD(modle.parameters(), lr=0.05, momentum=0.5)

def train(epoch):
    total_loss = 0
    for index, data in enumerate(train_load, 0):
        x_data, y_data = data
        y_prey = modle(x_data)
        loss = criteration(y_prey, y_data)
        total_loss += loss
        optimition.zero_grad()
        loss.backward()
        optimition.step()
        if index % 300 == 299:
            print(epoch + 1, index + 1, total_loss / 300)
            total_loss = 0

if __name__ == '__main__':
    for epoch in range(20):
        train(epoch)