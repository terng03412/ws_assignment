
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from PIL import Image
from .model import mobilenet_v2, Net, Net2


LEARNING_RATE = 1e-4
REGULARIZATION = 1e-6
BATCH_SIZE = 16
EPOCH = 10


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
        self._train = train

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

        if self._train:
            train_dir = '/code/dataset/dataset/train/'
            anchor_path = train_dir + \
                anchor_path.split('-')[0] + '/' + anchor_path
            positive_path = train_dir + \
                positive_path.split('-')[0] + '/' + positive_path
            negative_path = train_dir + \
                negative_path.split('-')[0] + '/' + negative_path
        else:
            test_dir = '/code/dataset/dataset/test/'
            anchor_path = test_dir + \
                anchor_path.split('-')[0] + '/' + anchor_path
            positive_path = test_dir + \
                positive_path.split('-')[0] + '/' + positive_path
            negative_path = test_dir + \
                negative_path.split('-')[0] + '/' + negative_path

        anchor_img = Image.open(anchor_path)
        positive_img = Image.open(positive_path)
        negative_img = Image.open(negative_path)

        if self.transform is not None:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img


def train():

    TRAIN_DATA_PATH = "/code/app/files/train_celeb.csv"
    TEST_DATA_PATH = "/code/app/files/test_celeb.csv"

    TRANSFORM_IMG = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    TRANSFORM_IMG = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor()
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = CelebDataset(csv_file=TRAIN_DATA_PATH,
                              train=True, transform=TRANSFORM_IMG)

    test_data = CelebDataset(csv_file=TEST_DATA_PATH,
                             train=False, transform=None)

    train_dataloader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)

    test_dataloader = DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)

    model = mobilenet_v2(32).to(device)
    model = Net(128).to(device)

    try:
        model = torch.load("/code/app/files/model.pt")
    except:
        print("Could not load model")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    min_loss = 100
    for epoch in range(1, EPOCH + 1):
        model.train()
        for batch_idx, (anchor_img, positive_img, negative_img) in enumerate(train_dataloader):
            anchor_img, positive_img, negative_img = anchor_img.to(
                device), positive_img.to(device), negative_img.to(device)
            optimizer.zero_grad()
            anchor = model(anchor_img)
            pos = model(positive_img)
            neg = model(negative_img)
            loss = triplet_loss(anchor, pos, neg)
            loss.backward()
            optimizer.step()

            if batch_idx % 20 == 0:
                print(
                    "Training : Epoch {} Iteration {}: Loss = {}".format(
                        epoch, batch_idx, loss
                    )
                )

        model.eval()

        for batch_idx, (anchor_img, positive_img, negative_img) in enumerate(test_dataloader):
            anchor_img, positive_img, negative_img = anchor_img.to(
                device), positive_img.to(device), negative_img.to(device)

            anchor = model(anchor_img)
            pos = model(positive_img)
            neg = model(negative_img)
            loss = triplet_loss(anchor, pos, neg)
            if (float(loss)) < min_loss:
                min_loss = (float(loss))
                PATH = "/code/app/files/model_" + \
                    str(epoch) + ".pt"
                torch.save(model, PATH)

            if batch_idx % 20 == 0:
                print(
                    "Testing : Epoch {} Iteration {}: Loss = {}: Min loss = {} ".format(
                        epoch, batch_idx, loss, min_loss
                    )
                )
