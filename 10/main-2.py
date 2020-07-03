#
# 这个程序最好不要跑，显卡会爆满，直接自动关机
#

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch import nn, optim

from resnet import ResNet18


def main():
    batch_size = 32

    cifar_train_data = datasets.CIFAR10('../dataset/CIFAR10', train=True, download=True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]))
    cifar_test_data = datasets.CIFAR10('../dataset/CIFAR10', train=False, download=True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]))

    # shuffle ==> 加载时需不需要随机化
    cifar_train = DataLoader(cifar_train_data, batch_size=batch_size, shuffle=True)
    cifar_test = DataLoader(cifar_test_data, batch_size=batch_size, shuffle=True)

    x, label = iter(cifar_train).next()
    print('x: ', x.shape, 'label: ', label.shape)

    device = torch.device('cuda')
    model = ResNet18().to(device)
    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(model)

    for epoch in range(1000):
        model.train()
        for batchidx, (x, label) in enumerate(cifar_train):
            # [b, 3, 32, 32]
            # [b]
            x, label = x.to(device), label.to(device)

            # logits 经过softmax ==> pred
            logits = model(x)
            # logits: [b, 10]
            # label: [b]
            # loss: tensor scalar
            loss = criteon(logits, label)

            # backprop
            # 将优化器中的梯度清零
            optimizer.zero_grad()
            # 梯度回归
            loss.backward()
            optimizer.step()

        #
        print("epoch: {}, loss: {}".format(epoch, loss.item()))

        # test
        model.eval()
        # 不需要 back propagate 构建计算图
        with torch.no_grad():
            total_correct = 0
            total_num = 0
            for x, label in cifar_test:
                x, label = x.to(device), label.to(device)

                # [b, 10]
                logits = model(x)
                # [b]
                pred = logits.argmax(dim=1)
                # [b] vs [b] => scalar tensor
                total_correct += torch.eq(pred, label).float().sum().item()
                total_num += x.size(0)

            acc = total_correct / total_num
            print("epoch: {}, acc: {}".format(epoch, acc))


if __name__ == '__main__':
    main()

