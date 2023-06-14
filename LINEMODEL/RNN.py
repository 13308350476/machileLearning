import torch

batch_size = 1
input_size = 4
seq_size = 5
hidden_size = 5
num_layer = 1

idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 2, 3]
y_data = [3, 1, 2, 3, 2] 

onehot_lookup = [[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]]

x_one_hot = [onehot_lookup[x] for x in x_data]
print(x_one_hot)
inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
print(inputs)
labels = torch.LongTensor(y_data).view(-1)
print(labels)


class Modle(torch.nn.Module):
    def __init__(self, batchsize, inputsize, hidden_size, num_layer=1):
        super(Modle, self).__init__()
        self.batch_size = batchsize
        self.input_size = inputsize
        self.hiden_size = hidden_size
        self.num_layer = num_layer
        self.rnn = torch.nn.RNN(input_size=self.input_size, hidden_size=self.hiden_size, num_layers=self.num_layer)

    def forward(self, x):
        hidden = torch.zeros(self.num_layer, self.batch_size, self.hiden_size)
        x,_ = self.rnn(x, hidden)
        print(x.view(-1, self.hiden_size))
        return x.view(-1, self.hiden_size)
    
modle = Modle(batch_size, input_size, hidden_size)

criterion = torch.nn.CrossEntropyLoss()
optimition = torch.optim.Adam(modle.parameters(), lr=0.1)

for epoch in range(1):
    hidden = modle(inputs)
    # print(hidden)
    # print(labels)
    loss = criterion(hidden, labels)
    # print(loss.item())
    optimition.zero_grad()
    loss.backward()
    optimition.step()
    print(hidden)
    TEMP, idx = hidden.max(dim=1)
    print(TEMP)
    print(idx)
    idx = idx.data.numpy()
    print(idx)

    print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
    print(', Epoch [%d/10] loss = %.3f' % (epoch + 1, loss.item()))