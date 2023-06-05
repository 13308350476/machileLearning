
import torchvision

train_set = torchvision.datasets.MNIST(root='/datasets/minist', train=True, download=True)
test_set = torchvision.datasets.MNIST(root='/datasets/minist', train=False, download=True)

