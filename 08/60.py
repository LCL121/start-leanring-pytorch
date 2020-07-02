import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import time


start_time = time.time()
batch_size = 200
learning_rate = 0.01
epochs = 10


train_db = torchvision.datasets.MNIST('../dataset', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ]))
test_db = torchvision.datasets.MNIST('../dataset', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ]))
print('train: ', len(train_db), 'test: ', len(test_db))
train_db, val_db = torch.utils.data.random_split(train_db, [50000, 10000])
print('db1: ', len(train_db), 'db2: ', len(val_db))


train_loader = torch.utils.data.DataLoader(train_db, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_db, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_db, batch_size=batch_size, shuffle=True)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.Dropout(0.5),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 200),
            nn.Dropout(0.5),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 10),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.model(x)
        return x


device = torch.device('cuda')
net = MLP().to(device)
optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.01, momentum=0.78)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4)
criteon = nn.CrossEntropyLoss().to(device)


for epoch in range(epochs):
    # train 的时候，可以dropout
    net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28*28)
        data, target = data.to(device), target.to(device)
        logits = net(data)
        loss = criteon(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                (len(data) * batch_idx / len(train_loader.dataset) * 100.), loss.item()
            ))
    val_loss = 0
    correct = 0
    for data, target in val_loader:
        data = data.view(-1, 28*28)
        data, target = data.to(device), target.to(device)
        logits = net(data)
        val_loss += criteon(logits, target).item()
        pred = logits.data.max(1)[1]
        correct += pred.eq(target.data).sum()
    val_loss /= len(val_loader.dataset)
    scheduler.step(val_loss)
    print('\nVAL set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)
    ))

# 在 test 的时候，要把dropout去掉
net.eval()
test_loss = 0
correct = 0
for data, target in test_loader:
    data = data.view(-1, 28*28)
    data, target = data.to(device), target.to(device)
    logits = net(data)
    test_loss += criteon(logits, target).item()
    pred = logits.data.max(1)[1]
    correct += pred.eq(target.data).sum()
test_loss /= len(test_loader.dataset)
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)
))


end_time = time.time()
print("Used time: {}".format(end_time - start_time))

