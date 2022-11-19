import os
import random
import pandas as pd

train_path = '/code/dataset/dataset/train/'
test_path = '/code/dataset/dataset/test/'
val_path = '/code/dataset/dataset/val/'

train = os.listdir(train_path)
test = os.listdir(test_path)
val = os.listdir(val_path)


def ranImg(images: list):
    l = len(images)
    r = random.randint(0, l-1)
    return images[r]


def create_train():
    anchor_list = []
    pos_list = []
    neg_list = []

    for i in range(len(train)-1):
        for j in range(i+1, len(train)):

            anchor_imgs = os.listdir(train_path+train[i])
            neg_imgs = os.listdir(train_path+train[j])
            pos_imgs = anchor_imgs

            anchor = ranImg(anchor_imgs)
            neg = ranImg(neg_imgs)
            pos = ranImg(pos_imgs)

            anchor_list.append(anchor)
            pos_list.append(pos)
            neg_list.append(neg)

    df = pd.DataFrame({'anchor': anchor_list,
                       'positive': pos_list,
                       'negative': neg_list})
    df.to_csv('/code/app/files/train_celeb.csv', index=False)


def create_test():
    anchor_list = []
    pos_list = []
    neg_list = []

    for i in range(len(test)-1):
        for j in range(i+1, len(test)):

            anchor_imgs = os.listdir(test_path+test[i])
            neg_imgs = os.listdir(test_path+test[j])
            pos_imgs = anchor_imgs

            anchor = ranImg(anchor_imgs)
            neg = ranImg(neg_imgs)
            pos = ranImg(pos_imgs)

            anchor_list.append(anchor)
            pos_list.append(pos)
            neg_list.append(neg)

    df = pd.DataFrame({'anchor': anchor_list,
                       'positive': pos_list,
                       'negative': neg_list})
    df.to_csv('/code/app/files/test_celeb.csv', index=False)


# create_test()
# create_train()
