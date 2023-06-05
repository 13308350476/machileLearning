import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from load_data import data_split, Excel_dataset
import time
from torch.utils.tensorboard import SummaryWriter


# Defining BP Neural Networks
class BPNerualNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(input_size, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, output_size),
                                   nn.LogSoftmax(dim=1)
                                   )

    def forward(self, x):
        x = self.model(x)

        return x


if __name__ == "__main__":
    data = Excel_dataset('C:/Users/tssh/Desktop/test.xlsx', if_normalize=False)
    data_train, data_test = data_split(data, 0.8)

    Epoch = 1000
    input_size = 4 #feture's number
    hidden_size = 12
    output_size = 2 #label's number
    LR = 0.005 #step lenth
    batchsize = 2

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = BPNerualNetwork()

    optimizer = torch.optim.Adam(net.parameters(), LR)

    # lost function
    loss_func = torch.nn.CrossEntropyLoss()

    # Train and record each accuracy rate
    data_loader = DataLoader(data_train, batch_size=batchsize, shuffle=False)
    a = time.time()

    writer = SummaryWriter('logs')
    for epoch in range(1001):
        # print(epoch)
        for step, data in enumerate(data_loader):
            net.train()
            inputs, labels = data
            # labels = labels.to(torch.float)
            # forword
            out = net(inputs)
            # loss func
            loss = loss_func(out, labels)
            # Clear the gradient from the previous round
            optimizer.zero_grad()
            # backword
            loss.backward()
            # update
            optimizer.step()

        if epoch % 10 == 0: #?????????
            net.eval()
            with torch.no_grad():
                total = len(data_test)
                yuce = 0
                test_dataloader = DataLoader(data_test, batch_size=1, shuffle=True)
                j = 0
                for i, (images, labels) in enumerate(test_dataloader):
                    # images = images.cuda(device)
                    # labels = labels.cuda(device)
                    outputs = net(images)
                    prediction = torch.max(outputs, 1)[1]  # torch.max

                    pred_y = prediction.numpy()  #It was placed on the GPU beforehand, so it must be taken from the GPU to the CPU!!!!
                    # print(pred_y,labels.data.numpy())

                    if pred_y[0] == labels.data.numpy()[0]:
                        j += 1

                    acc = j / total
                writer.add_scalar(f'LR={LR}\'s iris raght rate', acc, epoch + 1)

                print("tain number is", epoch + 1, "right rate is:", acc)

                # print(pred, labels)
    writer.close()
    print(time.time() - a)
