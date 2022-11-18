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
        model = models.resnext50_32x4d(pretrained=True)
        print("Pretrained model is loaded")
    else:
        model = models.resnext50_32x4d(pretrained=False)
    # Parameters of newly constructed modules have requires_grad=True by default
    model.fc = nn.Linear(model.fc.in_features, out_planes)
    return model


def mobilenet_v2(out_planes, pretrained=False):
    if pretrained is True:
        model = models.mobilenet_v2(pretrained=True)
        print("Pretrained model is loaded")
    else:
        model = models.mobilenet_v2(pretrained=False)

    # Parameters of newly constructed modules have requires_grad=True by default
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, out_planes)
    return model
