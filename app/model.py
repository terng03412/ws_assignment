import torch
import torch.nn as nn
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import models

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Training on [{}].'.format(device))


def resnext50(out_planes, pretrained=False):
    if pretrained is True:
        model = models.resnext50_32x4d(weights=True)
        print("Pretrained model is loaded")
    else:
        model = models.resnext50_32x4d(weights=None)
    # Parameters of newly constructed modules have requires_grad=True by default
    model.fc = nn.Linear(model.fc.in_features, out_planes)
    return model


def mobilenet_v2(out_planes, pretrained=False):
    if pretrained is True:
        model = models.mobilenet_v2(weights=True)
        print("Pretrained model is loaded")
    else:
        model = models.mobilenet_v2(weights=None)

    # Parameters of newly constructed modules have requires_grad=True by default
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, out_planes)
    return model

# Create a neural net class


class Net(nn.Module):

    # Defining the Constructor
    def __init__(self, num_classes=3):
        super(Net, self).__init__()

        # In the init function, we define each layer we will use in our model

        # Our images are RGB, so we have input channels = 3.
        # We will apply 12 filters in the first convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)

        # A second convolutional layer takes 12 input channels, and generates 24 outputs
        self.conv2 = nn.Conv2d(
            in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)

        # We in the end apply max pooling with a kernel size of 2
        self.pool = nn.MaxPool2d(kernel_size=2)

        # A drop layer deletes 20% of the features to help prevent overfitting
        self.drop = nn.Dropout2d(p=0.2)

        # Our 128x128 image tensors will be pooled twice with a kernel size of 2. 128/2/2 is 32.
        # This means that our feature tensors are now 32 x 32, and we've generated 24 of them

        # We need to flatten these in order to feed them to a fully-connected layer
        self.fc = nn.Linear(in_features=32 * 32 * 24, out_features=num_classes)

    def forward(self, x):
        # In the forward function, pass the data through the layers we defined in the init function

        # Use a ReLU activation function after layer 1 (convolution 1 and pool)
        x = F.relu(self.pool(self.conv1(x)))

        # Use a ReLU activation function after layer 2
        x = F.relu(self.pool(self.conv2(x)))

        # Select some features to drop to prevent overfitting (only drop during training)
        x = F.dropout(self.drop(x), training=self.training)

        # Flatten
        x = x.view(-1, 32 * 32 * 24)
        # Feed to fully-connected layer to predict class
        x = self.fc(x)
        # Return class probabilities via a log_softmax function
        return x


model = Net(128).to(device)
