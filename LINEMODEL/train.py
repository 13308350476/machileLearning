import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F


batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_sets = datasets.MNIST(root='../datasets/minist', train=True, download=False, transform=transform)
train_loader = DataLoader(dataset=train_sets, shuffle=True, batch_size=batch_size)

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 5, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(5, 10, kernel_size=4)
        self.conv3 = torch.nn.Conv2d(10, 20, kernel_size=2)
        self.pooling = torch.nn.MaxPool2d(2)
        self.linear1 = torch.nn.Linear(80, 60)
        self.linear2 = torch.nn.Linear(60, 40)
        self.linear3 = torch.nn.Linear(40, 10)

    def forward(self, x):
        batchsize = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = F.relu(self.pooling(self.conv3(x)))
        x = x.view(batchsize, -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    
modle = CNN()
criteration = torch.nn.CrossEntropyLoss()
optimition = torch.optim.SGD(modle.parameters(), lr=0.05, momentum=0.5)

def train(epoch):
    total_loss = 0
    for index, data in enumerate(train_loader, 0):
        x_data, y_data = data
        y_prey = modle(x_data)
        loss = criteration(y_prey, y_data)
        total_loss += loss
        optimition.zero_grad()
        loss.backward()
        optimition.step()
        if index % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, index + 1, total_loss / 300))
            total_loss = 0.0

if __name__ == '__main__':
    for epoch in range(1):
        train(epoch)