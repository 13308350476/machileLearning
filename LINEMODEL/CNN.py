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


train_set = datasets.MNIST(root='../datasets/minist', train=True, download=False, transform=transform)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

test_set = datasets.MNIST(root='../datasets/minist', train=False, download=False, transform=transform)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv_layer2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.linear = torch.nn.Linear(320, 10)

    def forward(self, x):
        batchsize = x.size(0)
        x = F.relu(self.pooling(self.conv_layer1(x)))
        x = F.relu(self.pooling(self.conv_layer2(x)))
        x = x.view(batchsize, -1)
        x = self.linear(x)
        return x

modle = CNN()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('cuda is aviluable:', torch.cuda.is_available())
modle.to(device)
criteration = torch.nn.CrossEntropyLoss()
optimition = torch.optim.SGD(modle.parameters(), lr=0.05, momentum=0.5)

def train(epoch):
    total_loss = 0
    for index, data in enumerate(train_loader, 0):
        x_data, y_data = data
        x_data, y_data = x_data.to(device), y_data.to(device)
        y_prey = modle(x_data)
        loss = criteration(y_prey, y_data)
        total_loss += loss
        optimition.zero_grad()
        loss.backward()
        optimition.step()
        if index % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, index + 1, total_loss / 300))
            total_loss = 0.0

# if __name__ == '__main__':
for epoch in range(1):
    train(epoch)