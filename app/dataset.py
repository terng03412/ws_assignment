from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn import preprocessing
# from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

import torch

from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn import metrics

import torch


class Dataset:

    def __init__(self):
        self.device = torch.device('cpu')

    def to(self, device):
        self.device = device
        return self

    def train_valid_test_split(self, train_size=0.7, valid_size=0.1, test_size=0.2):
        field_dims = (self.data.max(axis=0).astype(int) + 1).tolist()[:-1]

        train, valid_test = train_test_split(
            self.data, train_size=train_size, random_state=2021)

        valid_size = valid_size / (test_size + valid_size)
        valid, test = train_test_split(
            valid_test, train_size=valid_size, random_state=2021)

        device = self.device

        train_X = torch.tensor(train[:, :-1], dtype=torch.long).to(device)
        valid_X = torch.tensor(valid[:, :-1], dtype=torch.long).to(device)
        test_X = torch.tensor(test[:, :-1], dtype=torch.long).to(device)
        train_y = torch.tensor(
            train[:, -1], dtype=torch.float).unsqueeze(1).to(device)
        valid_y = torch.tensor(
            valid[:, -1], dtype=torch.float).unsqueeze(1).to(device)
        test_y = torch.tensor(
            test[:, -1], dtype=torch.float).unsqueeze(1).to(device)

        return field_dims, (train_X, train_y), (valid_X, valid_y), (test_X, test_y)


class LMWNDataset(Dataset):

    def __init__(self, file, read_part=True, sample_num=100000, task='classification'):
        super(LMWNDataset, self).__init__()

        dtype = {
            'user_id': np.int64,
            'restaurant_id': np.int64,
            'res_score': np.float16,
        }

        if read_part:
            data_df, df_norm = self.preprocess(file, sample_num=sample_num)
#             data_df = pd.read_csv(file, sep=',', dtype=dtype, nrows=sample_num)
#             data_df = pd.read_csv(file, sep=',', nrows=sample_num)

        else:
            data_df = pd.read_csv(file, sep=',', dtype=dtype)
        data_df = data_df.drop(columns=['time'])

        if task == 'classification':
            data_df['res_score'] = data_df.apply(
                lambda x: 1 if x['res_score'] > 0 else 0, axis=1).astype(np.int8)

        self.data = data_df.values
        self.df_norm = df_norm

    def get_norm_data(self):
        return self.df_norm

    def get_top_res(self):
        return self.df_norm.sort_values("res_score", ascending=False)[0:1000]

    def preprocess(self, file, sample_num):
        df = pd.read_csv(file, sep=',', nrows=sample_num, header=0, names=[
                         "Transaction_ID", "id", "user_id", "restaurant_id", "created_at"])

        res = set()
        ranked = dict()

        # Count restaurant ranked
        for i in df['restaurant_id']:
            res.add(i)
            ranked[i] = 0
        for i in df['restaurant_id']:
            ranked[i] += 1
        scores = []
        for i in df['restaurant_id']:
            scores.append(ranked[i])
        df['res_score'] = scores

        # Count how many time that user select this restaurant
        con = df['user_id']+df['restaurant_id']
        freqs = dict()
        for i in con:
            freqs[i] = -1
        for i in con:
            freqs[i] += 1
        user_freq = []
        for i in con:
            user_freq.append(freqs[i])
        df['freq'] = user_freq

        # Find history bought
        his = dict()
        for i in df['user_id']:
            his[i] = []
        for i in con:
            u_id = i[0:16]
            r_id = i[16:]
            his[u_id].append(r_id)
        user_his = []
        for i in df['user_id']:
            user_his.append(his[i])
        df['user_history'] = user_his

        # Convert time to only hour
        t = []
        for i in df['created_at']:
            t.append(int((i.split(' ')[1]).split(":")[0]))
        df['time'] = t

        # Normalize score
        # copy the data
        df_norm = df.copy()

        # apply normalization techniques
        column = 'res_score'
#         df_norm[column] = MinMaxScaler().fit_transform(np.array(df_norm[column] + 10*df_norm['freq']).reshape(-1,1))
        df_norm[column] = np.array(
            df_norm[column] + 5*df_norm['freq']).reshape(-1, 1)

        user_encoder = preprocessing.LabelEncoder()
        res_encoder = preprocessing.LabelEncoder()
        time_encoder = preprocessing.LabelEncoder()

        df_norm['UID'] = user_encoder.fit_transform(df_norm['user_id'].values)
        df_norm['RID'] = res_encoder.fit_transform(
            df_norm['restaurant_id'].values)
        df_norm['TID'] = time_encoder.fit_transform(df_norm['time'].values)

        self.user_encoder = user_encoder
        self.res_encoder = res_encoder

        return df_norm[['UID', 'RID', 'res_score', 'time']], df_norm

    def encoder(self):
        return self.user_encoder, self.res_encoder


def create_dataset(dataset='LMWN', read_part=True, sample_num=100000, task='classification', sequence_length=40, device=torch.device('cpu')):
    #     return LMWNDataset('./dataset/lmwn.csv', read_part=read_part, sample_num=sample_num, task=task).to(device)

    return LMWNDataset('files/transaction', read_part=read_part, sample_num=sample_num, task=task).to(device)


class EarlyStopper:

    def __init__(self, model, num_trials=50):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_metric = -1e9
        self.best_state = deepcopy(model.state_dict())
        self.model = model

    def is_continuable(self, metric):
        # maximize metric
        if metric > self.best_metric:
            self.best_metric = metric
            self.trial_counter = 0
            self.best_state = deepcopy(self.model.state_dict())
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


class BatchLoader:

    def __init__(self, X, y, batch_size=4096, shuffle=True):
        assert len(X) == len(y)

        self.batch_size = batch_size

        if shuffle:
            seq = list(range(len(X)))
            np.random.shuffle(seq)
            self.X = X[seq]
            self.y = y[seq]
        else:
            self.X = X
            self.y = y

    def __iter__(self):
        def iteration(X, y, batch_size):
            start = 0
            end = batch_size
            while start < len(X):
                yield X[start: end], y[start: end]
                start = end
                end += batch_size

        return iteration(self.X, self.y, self.batch_size)


class Trainer:

    def __init__(self, model, optimizer, criterion, batch_size=None, task='classification'):
        assert task in ['classification', 'regression']
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.task = task

    def train(self, train_X, train_y, epoch=100, trials=None, valid_X=None, valid_y=None):
        if self.batch_size:
            train_loader = BatchLoader(train_X, train_y, self.batch_size)
        else:
            # 为了在 for b_x, b_y in train_loader 的时候统一
            train_loader = [[train_X, train_y]]

        if trials:
            early_stopper = EarlyStopper(self.model, trials)

        train_loss_list = []
        valid_loss_list = []

        for e in tqdm(range(epoch)):
            # train part
            self.model.train()
            train_loss_ = 0
            for b_x, b_y in train_loader:
                self.optimizer.zero_grad()
                pred_y = self.model(b_x)
                train_loss = self.criterion(pred_y, b_y)
                train_loss.backward()
                self.optimizer.step()

                train_loss_ += train_loss.detach() * len(b_x)

            train_loss_list.append(train_loss_ / len(train_X))

            # valid part
            if trials:
                valid_loss, valid_metric = self.test(valid_X, valid_y)
                valid_loss_list.append(valid_loss)
                if not early_stopper.is_continuable(valid_metric):
                    break

        if trials:
            self.model.load_state_dict(early_stopper.best_state)
            plt.plot(valid_loss_list, label='valid_loss')

        plt.plot(train_loss_list, label='train_loss')
        plt.legend()
        plt.show()

        print('train_loss: {:.5f} | train_metric: {:.5f}'.format(
            *self.test(train_X, train_y)))

        if trials:
            print('valid_loss: {:.5f} | valid_metric: {:.5f}'.format(
                *self.test(valid_X, valid_y)))

    def test(self, test_X, test_y):
        self.model.eval()
        with torch.no_grad():
            pred_y = self.model(test_X)
#             print("shape : ", pred_y.shape, test_y.shape)
            test_loss = self.criterion(pred_y, test_y).detach()
        if self.task == 'classification':
            test_metric = metrics.roc_auc_score(test_y.cpu(), pred_y.cpu())
        elif self.task == 'regression':
            test_metric = -test_loss
        return test_loss, test_metric

    def inference(self, X):
        self.model.eval()
#         print(test_X.shape,X.shape)
        with torch.no_grad():
            pred_y = self.model(X)
        return pred_y

    def summary(self):
        print(self.model)

    def save(self, path):
        torch.save(self.model, path)
        print("save to ", path)

    def load(self, path):
        print('load from ', path)
        self.model = torch.load(path)
