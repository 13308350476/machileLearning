import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_sets = datasets.MNIST(root='../datasets/minist', train=True, download=False, transform=transform)
train_loader = DataLoader(dataset=train_sets, shuffle=True, batch_size=batch_size)


class Inception(nn.Module):
    def __init__(self, channel):
        super(Inception, self).__init__()
        self.aver_pool = nn.AvgPool2d(kernel_size=3, padding=1, stride=1)
        self.branch_pool = nn.Conv2d(channel, 24, kernel_size=1)

        self.branch1x1 = nn.Conv2d(channel, 16, kernel_size=1)

        self.branch5x5_1 = nn.Conv2d(channel, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch3x3_1 = nn.Conv2d(channel, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

    def forward(self, x):
        layer1 = self.aver_pool(x)
        layer1 = self.branch_pool(layer1)

        layer2 = self.branch1x1(x)
        
        layer3 = self.branch5x5_1(x)
        layer3 = self.branch5x5_2(layer3)

        layer4 = self.branch3x3_1(x)
        layer4 = self.branch3x3_2(layer4)
        layer4 = self.branch3x3_3(layer4)

        outputs = [layer1, layer2, layer3, layer4]
        return torch.cat(outputs, dim=1)


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.inceptin1 = Inception(channel=10)
        self.inceptin2 = Inception(channel=20)

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)

        self.pooling = nn.MaxPool2d(2)

        self.linear = nn.Linear(1408, 10)

        

    def forward(self, x):
        batchsize = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = self.inceptin1(x)
        x = F.relu(self.pooling(self.conv2(x)))
        x = self.inceptin2(x)
        x = x.view(batchsize, -1)
        x = self.linear(x)
        return x
    
modle = CNN()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'GPU')
modle.to(device)

criteration = torch.nn.CrossEntropyLoss()
optimition = torch.optim.SGD(modle.parameters(), lr=0.05, momentum=0.5)

def train(epoch):
    toltal_loss = 0
    for index, data in enumerate(train_loader, 0):
        x_data, y_data = data
        x_data, y_data = x_data.to(device), y_data.to(device)
        y_prey = modle(x_data)
        loss = criteration(y_prey, y_data)
        toltal_loss += loss
        optimition.zero_grad()
        loss.backward()
        optimition.step()
        if index % 300 == 299:
            print('[%d, %d] loss: %.3f' % (epoch + 1, index + 1, toltal_loss / 300))
            toltal_loss = 0.0

for epoch in range(1):
    train(epoch)