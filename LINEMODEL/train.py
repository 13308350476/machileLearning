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
        self.conv1 = torch.nn.Conv2d(1, 10, 3)
        self.conv2 = torch.nn.Conv2d(10, 20, 3)
        self.conv3 = torch.nn.Conv2d(20, 30, 3)
        self.pooling = torch.nn.MaxPool2d(2)
        self.l1 = torch.nn.Linear(30, 20)
        self.l2 = torch.nn.Linear(20, 10)
        

    def forward(self, x):
        batchsize = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = F.relu(self.pooling(self.conv3(x)))
        x = x.view(batchsize, -1)
        # print(x.shape)
        x = F.relu(self.l1(x))
        x = self.l2(x)
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