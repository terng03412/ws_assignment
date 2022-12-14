
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from PIL import Image
from model import mobilenet_v2, Net, Net2


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


def train(model, loss_func, device, train_loader, test_dataloader, optimizer, epoch):

    model.train()
    min_loss = 100
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

    STATE_PATH = "/code/app/files/model_mobilenet_state_" + \
        str(epoch) + ".pt"
    torch.save(model.state_dict(), STATE_PATH)

    STATE_PATH = "/code/dataset/model_mobilenet_state_" + \
        str(epoch) + ".pt"
    torch.save(model.state_dict(), STATE_PATH)

    model.eval()

    for batch_idx, (anchor_img, positive_img, negative_img) in enumerate(test_dataloader):
        anchor_img, positive_img, negative_img = anchor_img.to(
            device), positive_img.to(device), negative_img.to(device)

        anchor = model(anchor_img)
        pos = model(positive_img)
        neg = model(negative_img)
        loss = loss_func(anchor, pos, neg)

        if (float(loss)) < min_loss:
            min_loss = (float(loss))
            PATH = "/code/app/files/model_" + \
                str(epoch) + ".pt"
            torch.save(model, PATH)
            if min_loss == 0:
                min_loss += 1

        if batch_idx % 20 == 0:
            print(
                "Epoch {} Iteration {}: Loss = {}".format(
                    epoch, batch_idx, loss
                )
            )

    return model


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
print('Training on [{}].'.format(device))

train_data = CelebDataset(csv_file=TRAIN_DATA_PATH,
                          train=True, transform=TRANSFORM_IMG)

test_data = CelebDataset(csv_file=TEST_DATA_PATH,
                         train=False, transform=None)

train_dataloader = DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)

test_dataloader = DataLoader(
    test_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)

model = mobilenet_v2(32).to(device)
model = Net2(128).to(device)

try:
    model.load_state_dict(torch.load("/code/app/files/model_state.pt"))
except:
    print("Could not load model")

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)


for epoch in range(1, EPOCH + 1):
    model = train(model, triplet_loss, device,
                  train_dataloader, test_dataloader, optimizer, epoch)
