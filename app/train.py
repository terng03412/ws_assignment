
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torchvision import transforms

from pytorch_metric_learning import distances, losses, miners, reducers
import torchvision
from torch.utils import data

import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset

from PIL import Image


from model import mobilenet_v2


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Training on [{}].'.format(device))


LEARNING_RATE = 1e-4
REGULARIZATION = 1e-6
BATCH_SIZE = 4
EPOCH = 1


class CelebDataset(Dataset):

    def __init__(self, csv_file, train=False, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        self._num_of_classes = 526
        self._len_of_dataset = len(self.data)

    def get_num_of_classes(self):
        return self._num_of_classes

    def __len__(self):
        return self._len_of_dataset

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        anchor_path = self.data.iloc[idx, 0]
        positive_path = self.data.iloc[idx, 1]
        negative_path = self.data.iloc[idx, 2]

        if train:
            train_dir = '/code/dataset/dataset/train'
            anchor_path = train_dir + \
                anchor_path.split('-')[0] + '/' + anchor_path
            positive_path = train_dir + \
                positive_path.split('-')[0] + '/' + positive_path
            negative_path = train_dir + \
                negative_path.split('-')[0] + '/' + negative_path

        anchor_img = Image.open(anchor_path)
        positive_img = Image.open(positive_path)
        negative_img = Image.open(negative_path)

        if self.transform is not None:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img


def train(model, loss_func, device, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (anchor_img, positive_img, negative_img) in enumerate(train_loader):
        anchor_img, positive_img, negative_img = anchor_img.to(
            device), positive_img.to(device), negative_img.to(device)
        optimizer.zero_grad()
        anchor = model(anchor_img)
        pos = model(positive_img)
        neg = model(negative_img)
        loss = loss_func(anchor, pos, neg)
        loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            print(
                "Epoch {} Iteration {}: Loss = {}".format(
                    epoch, batch_idx, loss
                )
            )


# TRAIN_DATA_PATH = "/code/dataset/dataset/train"
# TEST_DATA_PATH = "/code/dataset/dataset/test"

TRAIN_DATA_PATH = "/content/drive/MyDrive/wisesight_data/train_celeb.csv"
TEST_DATA_PATH = "/content/drive/MyDrive/wisesight_data/test_celeb.csv"

TRANSFORM_IMG = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

train_data = CelebDataset(csv_file=TRAIN_DATA_PATH,
                          train=True, transform=TRANSFORM_IMG)
train_data_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)

model = mobilenet_v2(32).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 1


triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)


for epoch in range(1, num_epochs + 1):
    train(model, triplet_loss, device, train_data_loader, optimizer, epoch)
