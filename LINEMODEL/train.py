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

onehot = [onehot_lookup[x] for x in x_data]
inputs = torch.Tensor(onehot).view(-1, batch_size, input_size)

label = torch.LongTensor(y_data)

class NET(torch.nn.Module):
    def __init__(self, inputsize, batchsize, hiddensize, numlayer=1):
        super(NET, self).__init__()
        self.input_size = inputsize
        self.batch_size = batchsize
        self.hidden_size = hiddensize
        self.num_layer = numlayer
        self.rnn = torch.nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layer)

    def forward(self, x):
        hiddenlayer = torch.zeros(self.num_layer, self.batch_size, self.hidden_size)
        output,_ = self.rnn(x, hiddenlayer)
        return output.view(-1, self.hidden_size*self.batch_size)
    
modle = NET(input_size, batch_size, hidden_size)

criterion = torch.nn.CrossEntropyLoss()
optimition = torch.optim.Adam(modle.parameters(), lr=0.1)

for epoch in range(100):
    hidden = modle(inputs)
    loss = criterion(hidden, label)
    optimition.zero_grad()
    loss.backward()
    optimition.step()
    
    TEMP, idx = hidden.max(dim=1)
    # print(TEMP)
    # print(idx)
    idx = idx.data.numpy()
    print(idx)

    print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
    print(', Epoch [%d/10] loss = %.3f' % (epoch + 1, loss.item()))