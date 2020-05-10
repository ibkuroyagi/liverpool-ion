# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from wavenet import *
from torch.utils.data import Dataset, DataLoader
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
# from tqdm.notebook import tqdm
import gc
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import logging
import sys
import time

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)

# Any results you write to the current directory are saved as output.


# %%
# configurations and main hyperparammeters
EPOCHS = 150
NNBATCHSIZE = 16
GROUP_BATCH_SIZE = 4000
look_back = 1024
SEED = 321
LR = 0.001
SPLITS = 5
sample_batch = 10
sample_size = 500000

outdir = 'models'
flip = False
noise = False


if not os.path.exists(outdir):
    os.makedirs(outdir)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)


# %%
# read data
def read_data():
    train = pd.read_csv(
        'input/train.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels': np.int32})
    test = pd.read_csv(
        'input/test.csv', dtype={'time': np.float32, 'signal': np.float32})
    sub = pd.read_csv('input/sample_submission.csv',
                      dtype={'time': np.float32})
    return train, test, sub

# create batches of 4000 observations


def batching(df, batch_size):
    # print(df)
    df['group'] = df.groupby(df.index//batch_size,
                             sort=False)['signal'].agg(['ngroup']).values
    df['group'] = df['group'].astype(np.uint16)
    return df

# normalize the data (standard scaler). We can also try other scalers for a better score!


def normalize(train, test):
    train_input_mean = train.signal.mean()
    train_input_sigma = train.signal.std()
    train['signal'] = (train.signal - train_input_mean) / train_input_sigma
    test['signal'] = (test.signal - train_input_mean) / train_input_sigma
    train['signal'] = (((train.signal - train.signal.min()) /
                        (train.signal.max()-train.signal.min()))-0.5)*2
    test['signal'] = (((test.signal - test.signal.min()) /
                       (test.signal.max()-test.signal.min()))-0.5)*2
    return train, test


train, test, sample_submission = read_data()
train, test = normalize(train, test)


# %%


class subIronDataset(Dataset):
    def __init__(self, data, labels_df=None, training=True, index=None, mu=256, look_back=1024):
        self.data = data
        if training:
            #self.labels = pd.get_dummies(labels_df)
            self.labels = labels_df
        if index is not None:
            self.data = data.iloc[index]

        self.training = training
        self.index = index
        self.mu = mu
        self.len_sample = 500000  # １回の実験でサンプリングしたデータの点
        self.look_back = look_back
        self.class_num = 11

    def __len__(self):
        if self.index is not None:
            return len(self.index)
        else:
            return len(self.data)

    def __getitem__(self, idx):
        if self.index is not None:
            idx = self.index[idx]
        data = np.array([encode_mu_law(self.data[idx], mu=self.mu)])
        # labelを返す際にlook_back分のクラスを付与 label=(batch, look_back, class_num)
        # ただし、実装の簡単のためidxにはintが来る(dataloader)での想定とするlabel=(look_back, class_num)
        # look_backの部分は[now-look_back+1:now+1]という順番で入っている
        if self.training:
            # look_back未満の部分でmodeで保管する(もし精度が悪ければ変える)
            if idx % self.len_sample <= self.look_back-1:
                labels = np.zeros(self.look_back)
                new_idx = idx % self.len_sample
                num_batch = idx//self.len_sample
                labels[self.look_back-new_idx -
                       1:] = self.labels.iloc[self.len_sample*num_batch:idx+1].values
                labels[:self.look_back-new_idx-1] = labels[self.look_back -
                                                           new_idx-1:].sum(axis=0).argmax()
                #one_hot = np.zeros(self.class_num)
                # print(labels[self.look_back-idx-1:].sum(axis=0).argmax())
                # one_hot[labels[self.look_back-idx-1:].sum(axis=0).argmax()] = 1 #mode
                # print(one_hot)
                #labels[:self.look_back-idx-1] = one_hot
            else:  # look_back以降なので気にせず詰め込めばよい
                labels = self.labels.iloc[idx-self.look_back+1:idx+1].values
                # print(self.labels.iloc[idx-self.look_back+1:idx+1].values)
            #labels = self.labels.iloc[idx].values
            return [data, labels.astype("int64")]
        else:
            return data


# %%
NNBATCHSIZE = 200
look_back = 128
# train_dataset = IronDataset(train_df, train["open_channels"], training=True,look_back=look_back)
train_dataset = subIronDataset(
    train["signal"].values, train["open_channels"], training=True, look_back=look_back)
# test_dataset = IronDataset(test_df, training=False, look_back=look_back)
test_dataset = subIronDataset(
    test["signal"].values, training=False, look_back=look_back)
# idx = [0,4999999,499999,500000,500001,4999996]
# X, y = train_dataset[idx]
# train_dataloader = DataLoader(train_dataset, NNBATCHSIZE, shuffle=True, num_workers=8, pin_memory=True)
# test_dataloader = DataLoader(test_dataset, NNBATCHSIZE, shuffle=False, num_workers=8, pin_memory=True)
# -0.5805814
train_dataloader = DataLoader(
    train_dataset, NNBATCHSIZE, shuffle=True, num_workers=8, pin_memory=True)


# %%
#h = (B, n_aux, T)
# n_aux = 1
# model = WaveNet(n_quantize=256, n_aux=n_aux, n_resch=128, n_skipch=128,
#                  dilation_depth=9, dilation_repeat=2, kernel_size=2, upsampling_factor=0)
# print("model.receptive_field:",model.receptive_field)
# h = torch.zeros((200, n_aux, look_back))
# x,y = train_dataset[0]
# print(x.shape, y.shape)
# # print(x, y)
# for x, y in train_dataloader:
#     print(x.shape, y.shape)
#     pred = model(x, h)
#     print(pred.data.shape)
#     break


# %%
# print(pred.contiguous().view(-1,11).shape)
# print(y.contiguous().view(-1).shape)
# a = pred.contiguous().view(-1,11)
# # b = torch.LongTensor(y.contiguous().view(-1))
# b = y.contiguous().view(-1)
# print(a.shape, b.shape)
# print(a)
# print(b)
# criterion = nn.CrossEntropyLoss()
# loss = criterion(a,b)
# loss


# %%
class EarlyStopping:
    def __init__(self, patience=5, delta=0, checkpoint_path='checkpoint.pt', is_maximize=True):
        self.patience, self.delta, self.checkpoint_path = patience, delta, checkpoint_path
        self.counter, self.best_score = 0, None
        self.is_maximize = is_maximize

    def load_best_weights(self, model):
        model.load_state_dict(torch.load(self.checkpoint_path))

    def __call__(self, score, model):
        if self.best_score is None or (score > self.best_score + self.delta if self.is_maximize else score < self.best_score - self.delta):
            torch.save(model.state_dict(), self.checkpoint_path)
            self.best_score, self.counter = score, 0
            return 1
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return 2
        return 0

# %% [markdown]
# ## 学習の方法
#
# * マイ時刻のクロスエントロピーを計算する方法（本家）
# * 最終出力のみでクロスエントロピーを計算する（簡単）


# %%
devise = "cuda" if torch.cuda.is_available() else "cpu"
print(devise)


# %%
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
n_aux = 1
n_quantize = 256
look_back = 256
EPOCHS = 20
NNBATCHSIZE = 4000
lr = 1e-3
h = torch.zeros((NNBATCHSIZE, n_aux, look_back)).to(devise)

oof_score = []
for index, (train_index, val_index) in enumerate(kf.split(np.zeros((train.shape[0], 1)), train["open_channels"])):
    train_dataset = subIronDataset(
        train["signal"], train["open_channels"], training=True, index=train_index, look_back=look_back)
    train_dataloader = DataLoader(
        train_dataset, NNBATCHSIZE, shuffle=True, num_workers=8, pin_memory=True)

    valid_dataset = subIronDataset(
        train["signal"], train["open_channels"], training=True, index=val_index, look_back=look_back)
    valid_dataloader = DataLoader(
        valid_dataset, NNBATCHSIZE, shuffle=False, num_workers=4, pin_memory=True)

    it = 0

    model = WaveNet(n_quantize=256, n_aux=n_aux, n_resch=64, n_skipch=128,
                    dilation_depth=6, dilation_repeat=2, kernel_size=2, upsampling_factor=0)
    model = model.to(devise)

    early_stopping = EarlyStopping(patience=10, is_maximize=True,
                                   checkpoint_path=os.path.join("models", "checkpoint_fold_{}_iter_{}.pt".format(index, it)))
    cols = ["loss", "F1", "val_loss", "val_F1", "lr"]
    results = pd.DataFrame(columns=cols)
    weight = None  # cal_weights()
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.2)
    avg_train_losses, avg_valid_losses = [], []

    for epoch in tqdm(range(EPOCHS)):
        print('**********************************')
        print("Folder : {} Epoch : {}".format(index, epoch))
        print("Curr learning_rate: {:0.9f}".format(
            optimizer.param_groups[0]['lr']))
        train_losses, valid_losses = [], []
        tr_loss_cls_item, val_loss_cls_item = [], []

        model.train()  # prep model for training
        train_preds, train_true = torch.Tensor([]).to(
            devise), torch.LongTensor([]).to(devise)
        cnt = 0
        for x, y in tqdm(train_dataloader):
            x = x.to(devise)
            y = y.to(devise)

            optimizer.zero_grad()
            #loss_fn(model(input), target).backward()
            # optimizer.zero_grad()
            predictions = model(x, h)
            if cnt == 0:
                print("non pred", predictions.shape)
                print("non y   ", y.shape)
            loss = criterion(predictions.contiguous(
            ).view(-1, 11), y.contiguous().view(-1))

            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()

            # schedular.step()
            # record training lossa
            train_losses.append(loss.item())
            train_true = torch.cat([train_true, y.contiguous()[:, -1]], 0)
            train_preds = torch.cat(
                [train_preds, predictions.contiguous()[:, -1, :]], 0)
            cnt += 1
        model.eval()  # prep model for evaluation
        # optimizer.swap_swa_sgd()
        val_preds, val_true = torch.Tensor([]).to(
            devise), torch.LongTensor([]).to(devise)
        print('EVALUATION')
        with torch.no_grad():
            for x, y in valid_dataloader:
                x = x.to(devise)
                y = y.to(devise)

                predictions = model(x, h)
                loss = criterion(predictions.contiguous(
                ).view(-1, 11), y.contiguous().view(-1))
                valid_losses.append(loss.item())

                val_true = torch.cat([val_true, y.contiguous()[:, -1]], 0)
                val_preds = torch.cat(
                    [val_preds, predictions.contiguous()[:, -1, :]], 0)

        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        print("train_loss: {:0.6f}, valid_loss: {:0.6f}".format(
            train_loss, valid_loss))

        train_score = f1_score(train_true.cpu().detach().numpy(), train_preds.cpu().detach().numpy().argmax(1),
                               labels=list(range(11)), average='macro')

        val_score = f1_score(val_true.cpu().detach().numpy(), val_preds.cpu().detach().numpy().argmax(1),
                             labels=list(range(11)), average='macro')
        tmp = pd.DataFrame([train_loss, train_score, valid_loss,
                            val_score, optimizer.param_groups[0]['lr']], columns=cols)
        results = pd.concat([results, tmp], axis=0)
        results.to_csv('output/results_fold{}.csv'.format(index), index=False)
        schedular.step(val_score)
        print("train_f1: {:0.6f}, valid_f1: {:0.6f}".format(
            train_score, val_score))
        res = early_stopping(val_score, model)
        #print('fres:', res)
        if res == 2:
            print("Early Stopping")
            print('folder %d global best val max f1 model score %f' %
                  (index, early_stopping.best_score))
            break
        elif res == 1:
            print('save folder %d global val max f1 model score %f' %
                  (index, val_score))
    print('Folder {} finally best global max f1 score is {}'.format(
        index, early_stopping.best_score))
    oof_score.append(round(early_stopping.best_score, 6))
    break


# %%
test_dataset = subIronDataset(test["signal"], training=False)
test_dataloader = DataLoader(test_dataset, NNBATCHSIZE, shuffle=False)
model.eval()
pred_list = []
with torch.no_grad():
    for x, y in test_dataloader:
        x = x.to(devise)
        y = y.to(devise)

        predictions = model(x, h)
        pred_list.append(F.softmax(predictions.contiguous()[
                         :, -1, :], dim=1).cpu().numpy())  # shape (512000, 11)
    test_preds = np.vstack(pred_list)  # shape [2000000, 11]
    test_preds_all += test_preds
print('all folder score is:%s' % str(oof_score))
print('OOF mean score is: %f' % (sum(oof_score)/len(oof_score)))
print('Generate submission.............')
test_preds_all = test_preds_all / np.sum(test_preds_all, axis=1)[:, None]
test_pred_frame = pd.DataFrame({'time': sample_submission['time'].astype(str),
                                'open_channels': np.argmax(test_preds_all, axis=1)})
test_pred_frame.to_csv("preds/wavenet_preds.csv", index=False)
print('over')
