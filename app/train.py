
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torchvision import transforms

from pytorch_metric_learning import distances, losses, miners, reducers
import torchvision
from torch.utils import data

from model import mobilenet_v2


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Training on [{}].'.format(device))


LEARNING_RATE = 1e-4
REGULARIZATION = 1e-6
BATCH_SIZE = 32
EPOCH = 100


def train(model, loss_func, mining_func, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(
                "Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(
                    epoch, batch_idx, loss, mining_func.num_triplets
                )
            )


device = torch.device("cuda")

transform = transforms.Compose(
    [transforms.ToTensor()]
)

TRAIN_DATA_PATH = "/code/dataset/dataset/train"
TEST_DATA_PATH = "/code/dataset/dataset/test"

TRANSFORM_IMG = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

train_data = torchvision.datasets.ImageFolder(
    root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
train_data_loader = data.DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
test_data = torchvision.datasets.ImageFolder(
    root=TEST_DATA_PATH, transform=TRANSFORM_IMG)
test_data_loader = data.DataLoader(
    test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)


model = mobilenet_v2(64).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20


### pytorch-metric-learning stuff ###
distance = distances.CosineSimilarity()
reducer = reducers.ThresholdReducer(low=0)
loss_func = losses.TripletMarginLoss(
    margin=0.2, distance=distance, reducer=reducer)
mining_func = miners.TripletMarginMiner(
    margin=0.2, distance=distance, type_of_triplets="semihard"
)


for epoch in range(1, num_epochs + 1):
    train(model, loss_func, mining_func, device,
          train_data_loader, optimizer, epoch)
    # test(dataset1, dataset2, model, accuracy_calculator)
