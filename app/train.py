from dataset import create_dataset, Trainer
import torch
import torch.nn as nn
import torch.optim as optim
from model import AttentionalFactorizationMachine
import numpy as np


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Training on [{}].'.format(device))
dataset = create_dataset('lmwn', sample_num=100000,
                         device=device, task="regression")
field_dims, (train_X, train_y), (valid_X, valid_y), (test_X,
                                                     test_y) = dataset.train_valid_test_split()


EMBEDDING_DIM = 8
LEARNING_RATE = 1e-4
REGULARIZATION = 1e-6
BATCH_SIZE = 4096
EPOCH = 800
TRIAL = 100


afm = AttentionalFactorizationMachine(field_dims, EMBEDDING_DIM).to(device)
if torch.cuda.device_count() > 1:
    afm = nn.DataParallel(afm)
# In `DataParallel` mode, it's to specify the leader to gather parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
afm = afm.to(device)

optimizer = optim.Adam(afm.parameters(), lr=LEARNING_RATE,
                       weight_decay=REGULARIZATION)
# criterion = nn.BCELoss()

criterion = nn.MSELoss()

trainer = Trainer(afm, optimizer, criterion, BATCH_SIZE, task='regression')
trainer.summary()


trainer.train(train_X, train_y, epoch=EPOCH, trials=TRIAL,
              valid_X=valid_X, valid_y=valid_y)
test_loss, test_auc = trainer.test(test_X, test_y)
print('test_loss:  {:.5f} | test_auc:  {:.5f}'.format(test_loss, test_auc))


trainer.save('./files/model.h5')


# for inference
# user_encoder, res_encoder = dataset.encoder()
# example
# user_id = '549BB67D90E7982B'
# res_id = '2CBF3C995B3EA080'
# score = inference(user_encoder, res_encoder, trainer, user_id, res_id)
# print(score)

df_norm = dataset.get_norm_data()

user_encoder, res_encoder = dataset.encoder()
top_res = dataset.get_top_res()
top_res_RID = list(set(top_res['RID']))
user_topk = dict()

for i in range(df_norm['UID'].max()):
    user_pred_combi = []
    for j in top_res_RID:
        user_pred_combi.append([i, j])

    res = trainer.inference(torch.tensor(user_pred_combi))
    res = res.reshape(res.shape[0])
    _, indexes = torch.topk(res, 10)
    top_k_res = res_encoder.inverse_transform(
        [top_res_RID[i] for i in indexes.tolist()])
    user_topk[user_encoder.inverse_transform([i])[0]] = list(top_k_res)


np.save('./files/topk.npy', user_topk)

new_dict = np.load('./files/topk.npy', allow_pickle='TRUE').item()

print(new_dict['0003E1FDB847FCE8'])
