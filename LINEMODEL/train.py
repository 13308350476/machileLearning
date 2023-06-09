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


train_set = datasets.MNIST(root='../datasets/minist', train=True, Download=False, transform=transform)
train_load = DataLoader(dataset=train_set, shuffle=True, batch_size=batch_size)

test_set = datasets.MNIST(root='../datasets/minist', train=False, Download=False, transform=transform)
test_load = DataLoader(dataset=test_set, shuffle=False, batch_size=batch_size)

class NET(torch.nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        self.l1 = torch.nn.Linear(784, 128)
        self.l2 = torch.nn.Linear(128, 10)

