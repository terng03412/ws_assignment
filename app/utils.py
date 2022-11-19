# PREDICT
import os
import numpy as np
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
from model import mobilenet_v2, Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM_IMG = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor()
])


def embedded(model, image_path):

    image = Image.open(image_path)
    image = TRANSFORM_IMG(image)
    image = image.to(device)

    with torch.no_grad():
        output = model(image)

    return output


def create_embedded(output_path, dir_path, model):
    embedded_dict = dict()
    train_files = os.listdir(dir_path)
    for f in train_files:
        class_name = f
        images = os.listdir(dir_path+f)
        path = dir_path + '/' + str(f) + '/'
        for i in images:
            image_path = path + str(i)
            embedded_dict[i] = embedded(model, image_path).cpu().numpy()

    np.save(os.path.join(output_path, "embedded_all.npy"), embedded_dict)


def load_embedded_dict(embedded_dict_path):
    data = np.load(embedded_dict_path, allow_pickle=True)
    embedded_dict = dict(enumerate(data.flatten()))
    return embedded_dict[0]


def cal_similarity(distance_dict, image_path, model):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    TRANSFORM_IMG = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor()
    ])

    image = Image.open(image_path)
    image = TRANSFORM_IMG(image)
    image = image.to(device)
    output = None
    with torch.no_grad():
        output = model(image)

    output = output.cpu()
    distance = dict()
    count = dict()

    for i in distance_dict:
        name = i.split('-')[0]
        distance[name] = 0
        count[name] = 0

    for i in distance_dict:
        name = i.split('-')[0]
        ref = torch.from_numpy(distance_dict[i])
        dis = cos(output, ref)
        distance[name] += float(dis)
        count[name] += 1.0

    for i in distance_dict:
        distance[name] = distance[name]/count[name]

    return distance


model = Net(128)

STATE_PATH = "/code/app/files/model_state.pt"
model.load_state_dict(torch.load(STATE_PATH, map_location=torch.device('cpu')))
model.to(device)
print('load model')


out_p = '/code/app/files/'
dir_path = '/code/dataset/dataset/test/'
print('create_embedded')
create_embedded(out_p, dir_path, model)

embedded = load_embedded_dict(out_p + 'embedded_all.npy')
test_image = '/code/dataset/dataset/val/10108319/10108319-00015.jpg'

print('cal_similarity')
d = cal_similarity(embedded, test_image, model)
print(min(d, key=d.get), max(d, key=d.get))
