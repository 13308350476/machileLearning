import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_set = datasets.MNIST(root='../datasets/minist', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)

test_set = datasets.MNIST(root='/datasets/minist', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, shuffle=False, batch_size=1)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)

module = Net()

criteration = torch.nn.CrossEntropyLoss()
optimition = optim.SGD(module.parameters(), lr = 0.05, momentum=0.5)

def train(epoch):
    running_loss = 0.0
    for index, data in enumerate(train_loader, 0):
        x_data, y_data = data
        y_prey = module(x_data)
        loss = criteration(y_prey, y_data)
        optimition.zero_grad()
        loss.backward()
        optimition.step()
        running_loss += loss
        if index % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, index + 1, running_loss / 300))
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    i = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = module(images)
            predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels)
            if i == 0:
                print('batch size:', labels.size(0))
                i = 1
                print('predicted:', predicted)
                print('labels',labels)
    print('Accuracy on test set: %d %%' % (100 * correct / total))
    print('total:', total)


if __name__ == '__main__':
    for epoch in range(1):
        train(epoch)
        test()

