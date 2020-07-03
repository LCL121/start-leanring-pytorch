import torch
import torchvision


batch_size = 200


train_data = torchvision.datasets.MNIST('../dataset', train=True, download=True,
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.RandomHorizontalFlip(),
                                            torchvision.transforms.RandomVerticalFlip(),
                                            # -15 deg ~ 15 deg
                                            torchvision.transforms.RandomRotation(15),
                                            # 90 deg ~ 270 deg
                                            torchvision.transforms.RandomRotation([90, 270]),
                                            torchvision.transforms.Resize([32, 32]),
                                            torchvision.transforms.RandomCrop([28, 28]),
                                            torchvision.transforms.ToTensor()
                                        ]))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

print(train_data.data.shape)
