import csv
import glob
import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class Pokemon(Dataset):

    def __init__(self, root, resize, mode):
        super(Pokemon, self).__init__()

        self.root = root
        self.resize = resize

        self.name2label = {}  # "name":label
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue

            self.name2label[name] = len(self.name2label.keys())
        # print(self.name2label)

        # image, label
        self.images, self.labels = self.load_csv('image.csv')

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

        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.name2label.keys():
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))
                images += glob.glob(os.path.join(self.root, name, '*.gif'))

            # 1168, ['pokemon\\bulbasaur\\00000000.png'...]
            # print(len(images), images)

            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    # 'pokemon\\bulbasaur\\00000000.png', 0
                    writer.writerow([img, label])
                print('writen into csv file:', filename)

        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
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

    def denormalize(self, x_hot):

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # x_hot = (x - mean) / std
        # x = x_hot * std = mean
        # x: [c, h, w]
        # mean: [3] => [3, 1, 1]
        # print('pre_mean:', torch.tensor(mean).shape, 'pre_std:', torch.tensor(std).shape)
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        # print('mean:', mean.shape, 'std:', std.shape, 'x_hot:', x_hot.shape)

        x = x_hot * std + mean

        return x

    def __getitem__(self, idx):
        # idx:[0 ~ len(images)]
        # self.images, self.labels
        # img: 'pokemon\\squirtle\\00000205.jpg'
        # label: 0
        img, label = self.images[idx], self.labels[idx]
        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
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

    viz = visdom.Visdom()

    db = Pokemon('pokemon', 64, 'train')

    x, y = iter(db).__next__()
    # print('sample:', x.shape, y.shape, y)
    viz.image(db.denormalize(x), win='sample_x', opts=dict(title='sample_x'))

    loader = DataLoader(db, batch_size=32, shuffle=True)
    for x, y in loader:
        viz.images(db.denormalize(x), nrow=8, win='batch', opts=dict(tital='batch'))
        viz.text(str(y.numpy()), win='label', opts=dict(tital='batch_y'))

        time.sleep(10)

if __name__ == '__main__':
    main()
