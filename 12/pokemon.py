import torch
import os, glob
import random, csv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class Pokemon(Dataset):

    def __init__(self, root, resize, mode):
        super(Pokemon, self).__init__()

        self.root = root
        self.resize = resize

        self.name2label = {}
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue

            self.name2label[name] = len(self.name2label.keys())

        # print(self.name2label)

        # image, label
        self.images, self.labels = self.load_csv('images.csv')

        if mode == 'train':
            self.images = self.images[:int(0.6 * len(self.images))]
            self.labels = self.labels[:int(0.6 * len(self.labels))]
        elif mode == 'val':
            self.images = self.images[int(0.6 * len(self.images)):int(0.8 * len(self.images))]
            self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8 * len(self.labels))]
        else:
            self.images = self.images[int(0.8 * len(self.images)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]

    def load_csv(self, filename):
        # write
        if not os.path.exists('./{}'.format(filename)):
            images = []
            for name in self.name2label.keys():
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))
                images += glob.glob(os.path.join(self.root, name, '*.gif'))

            # ../dataset/pokemon/bulbasaur/00000228.png
            # print(len(images), images)

            random.shuffle(images)
            with open('./{}'.format(filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    writer.writerow([img, label])
                print('writen into csv file: ', filename)
        # read
        images, labels = [], []
        with open('./{}'.format(filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label)
                images.append(img)
                labels.append(label)
        assert len(images) == len(labels)

        return images, labels

    def __len__(self):
        return len(self.images)

    def denormalize(self, x_hat):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # x_hot = (x - mean) / std
        # x = x_hat * std + mean
        # x: [c, h, w]
        # mean: [3] => [3, 1, 1]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        x = x_hat * std + mean

        return x


    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]

        tf = transforms.Compose([
            # string path => image data
            lambda x:Image.open(x).convert('RGB'),
            transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        img = tf(img)
        label = torch.tensor(label)

        return img, label


def main():

    import visdom
    import time
    import torchvision

    viz = visdom.Visdom()

    # 图片很规整可以怎么简单的写
    # tf = transforms.Compose([
    #     transforms.Resize((64, 64)),
    #     transforms.ToTensor()
    # ])
    # db = torchvision.datasets.ImageFolder(root='../dataset/pokemon', transform=tf)
    # loader = DataLoader(db, batch_size=32, shuffle=True)
    #
    # for x, y in loader:
    #     viz.images(x, nrow=8, win='dataloader', opts=dict(title='dataloader_title'))
    #     viz.text(str(y.numpy()), win='label', opts=dict(title='label_title'))
    #     time.sleep(10)

    db = Pokemon('../dataset/pokemon', 224, 'train')
    x, y = next(iter(db))
    print('sample: ', x.shape, y)
    viz.images(db.denormalize(x), win='sample_x', opts=dict(title='sample_title'))
    loader = DataLoader(db, batch_size=32, shuffle=True, num_workers=8)

    for x, y in loader:
        viz.images(db.denormalize(x), nrow=8, win='dataloader', opts=dict(title='dataloader_title'))
        viz.text(str(y.numpy()), win='label', opts=dict(title='label_title'))
        time.sleep(10)


if __name__ == '__main__':
    main()


