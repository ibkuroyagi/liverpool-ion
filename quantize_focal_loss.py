# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
import os
from tqdm import tqdm
#from tqdm.notebook import tqdm
import gc
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import warnings
from torch.utils.data import Dataset, DataLoader
from wavenet import *
from wavenet2 import *
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)

# Any results you write to the current directory are saved as output.


# %%
# configurations and main hyperparammeters
EPOCHS = 300
NNBATCHSIZE = 32
GROUP_BATCH_SIZE = 5000
SEED = 42
LR = 0.001
SPLITS = 5

outdir = 'wavenet_models'
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
# it=21 小さいやつ
# it=31 大きいやつ
it = 31
print("it:", it)

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
    train = pd.read_csv('input/data-without-drift/train_clean.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})
    test  = pd.read_csv('input/data-without-drift/test_clean.csv', dtype={'time': np.float32, 'signal': np.float32})
    #from https://www.kaggle.com/sggpls/wavenet-with-shifted-rfc-proba and
    # https://www.kaggle.com/c/liverpool-ion-switching/discussion/144645
    Y_train_proba = np.load("input/ion-shifted-rfc-proba/Y_train_proba.npy")
    Y_test_proba = np.load("input/ion-shifted-rfc-proba/Y_test_proba.npy")
    for i in range(11):
        train[f"proba_{i}"] = Y_train_proba[:, i]
        test[f"proba_{i}"] = Y_test_proba[:, i]

    #sub  = pd.read_csv('input/sample_submission.csv', dtype={'time': np.float32})
    return train, test

# create batches of 4000 observations
def batching(df, batch_size):
    #print(df)
    df['group'] = df.groupby(df.index//batch_size, sort=False)['signal'].agg(['ngroup']).values
    df['group'] = df['group'].astype(np.uint16)
    return df

# normalize the data (standard scaler). We can also try other scalers for a better score!
def normalize(train, test):
    train_input_mean = train.signal.mean()
    train_input_sigma = train.signal.std()
    # train['signal'] = (train.signal - train_input_mean) / (3*train_input_sigma)
    # test['signal'] = (test.signal - train_input_mean) / (3*train_input_sigma)
    train['signal'] = (train.signal - train_input_mean + 2.05) / (4.15*train_input_sigma)
    test['signal'] = (test.signal - train_input_mean + 2.05) / (4.15*train_input_sigma)
    return train, test

# get lead and lags features
def lag_with_pct_change(df, windows):
    for window in windows:    
        df['signal_shift_pos_' + str(window)] = df.groupby('group')['signal'].shift(window).fillna(0)
        df['signal_shift_neg_' + str(window)] = df.groupby('group')['signal'].shift(-1 * window).fillna(0)
    return df

# main module to run feature engineering. Here you may want to try and add other features and check if your score imporves :).
def run_feat_engineering(df, batch_size):
    # create batches
    df = batching(df, batch_size = batch_size)
    # create leads and lags (1, 2, 3 making them 6 features)
    df = lag_with_pct_change(df, [1, 2, 3])
    # create signal ** 2 (this is the new feature)
    df['signal_2'] = df['signal'] ** 2
    # df['signal_3'] = df['signal'] ** 3
    # df["signal_2_log"] = np.log10(df['signal_2'])/ 11.5
    df['encode'] = encode_mu_law(df['signal'].values)
    return df

# fillna with the mean and select features for training
def feature_selection(test):
    features = [col for col in test.columns if col not in ['index', 'group', 'open_channels', 'time']]
    test = test.replace([np.inf, -np.inf], np.nan)
    for feature in features:
        feature_mean = test[feature].mean()
        test[feature] = test[feature].fillna(feature_mean)
    return test, features


def split(GROUP_BATCH_SIZE=4000):
    print('Reading Data Started...')
    train, test = read_data()
    train, test = normalize(train, test)
    test = run_feat_engineering(test, batch_size=GROUP_BATCH_SIZE)
    test, features = feature_selection(test)
    test = np.array(list(test.groupby('group').apply(lambda x: x[features].values)))
    return test

def normalize1(train, test):
    train_input_mean = train.signal.mean()
    train_input_sigma = train.signal.std()
    train['signal'] = (train.signal - train_input_mean + 1.75) / (4.2*train_input_sigma)
    test['signal'] = (test.signal - train_input_mean + 1.75) / (4.2*train_input_sigma)
    return train, test

def normalize2(train, test):
    train_input_mean = train.signal.mean()
    train_input_sigma = train.signal.std()
    train['signal'] = (train.signal - train_input_mean + 1.1) / (4.*train_input_sigma)
    test['signal'] = (test.signal - train_input_mean + 1.1) / (4.*train_input_sigma)
    return train, test

# %%
class IronDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.data[idx]
        onehot = np.eye(256)[data[:, -1].astype(int)]
        data = np.delete(data, -1, 1)
        data = np.concatenate([data, onehot], axis=1)
        data = data.transpose(1, 0)
        labels = self.labels[idx]

        return [data.astype(np.float32), labels.astype(int)]


class FocalLossWithOutOneHot(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLossWithOutOneHot, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)
        logit = logit.clamp(self.eps, 1. - self.eps)
        logit_ls = torch.log(logit)
        loss = F.nll_loss(logit_ls, target, reduction="none")
        view = target.size() + (1,)
        index = target.view(*view)
        loss = loss * (1 - logit.gather(1, index).squeeze(1)) ** self.gamma # focal loss

        return loss.sum()


def make_dataset(train, test, slide=2000):
    ex_size = 500000
    n_iter = int(train.shape[0]/ex_size)
    #test = run_feat_engineering(test, batch_size=GROUP_BATCH_SIZE)
    for i in range(n_iter):
        if slide == 0:
            train_i = train.iloc[i*ex_size + slide: (i+1)*ex_size + slide].reset_index(drop=True)
        else:
            train_i = train.iloc[i*ex_size + slide: (i+1)*ex_size + slide - GROUP_BATCH_SIZE].reset_index(drop=True)
        train_i = run_feat_engineering(train_i, batch_size=GROUP_BATCH_SIZE)
        train_i, features = feature_selection(train_i)
        target_cols = ['open_channels']
        train_tr_i = np.array(list(train_i.groupby('group').apply(lambda x: x[target_cols].values))).astype(np.float32)
        train_i = np.array(list(train_i.groupby('group').apply(lambda x: x[features].values)))
        if i == 0:
            X = train_i
            y = train_tr_i
        else:
            X = np.concatenate([X, train_i], 0)
            y = np.concatenate([y, train_tr_i], 0)
    return X, y


# %%
train, test = read_data()
# data split
if it // 10 == 2:
    train_idx = np.ones(5000000).astype(bool)
    train_idx[500000*4:500000*5] = False
    #train_idx[500000*9:500000*10] = False
    train = train[train_idx].reset_index(drop=True)
    train, test = normalize1(train, test)
    # tmp, tmp_tr = make_dataset(train, test, slide=2000)
    train, train_tr = make_dataset(train, test, slide=0)
    # train = np.concatenate([train, tmp], 0)
    # train_tr = np.concatenate([train_tr, tmp_tr], 0)
    # del tmp, tmp_tr
    # gc.collect()
elif it // 10 == 3:
    train_idx = np.zeros(5000000).astype(bool)
    train_idx[500000*4:500000*6] = True
    train_idx[500000*8:500000*10] = True
    train = train[train_idx].reset_index(drop=True)
    train, test = normalize2(train, test)
    train, train_tr = make_dataset(train, test, slide=0)

# %%
# test = split(GROUP_BATCH_SIZE=GROUP_BATCH_SIZE)
test = run_feat_engineering(test, batch_size=GROUP_BATCH_SIZE)
test, features = feature_selection(test)
test = np.array(list(test.groupby('group').apply(lambda x: x[features].values)))
print(train.shape, train_tr.shape)
print(test.shape)

# %%
test_y = np.zeros([int(2000000/GROUP_BATCH_SIZE), GROUP_BATCH_SIZE, 1])
test_dataset = IronDataset(test, test_y)
test_dataloader = DataLoader(test_dataset, NNBATCHSIZE, shuffle=False, num_workers=8, pin_memory=True)
test_preds_all = np.zeros((2000000, 11))

kf = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
val_score = 0
oof_score = []
for index, (train_index, val_index) in enumerate(kf.split(train, train_tr[:, -1, 0])):
    print("Fold : {}".format(index))
    train_dataset = IronDataset(train[train_index], train_tr[train_index])
    train_dataloader = DataLoader(train_dataset, NNBATCHSIZE, shuffle=True, num_workers=8, pin_memory=True)

    valid_dataset = IronDataset(train[val_index], train_tr[val_index])
    valid_dataloader = DataLoader(valid_dataset, NNBATCHSIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = WaveNet0(n_quantize=128, n_feature=train.shape[2]+255, n_resch=64, n_skipch=64,
                     dilation_depth=10, dilation_repeat=3, kernel_size=2)
    model = model.to(device)

    early_stopping = EarlyStopping(patience=40, is_maximize=True, device=device,
                                   checkpoint_path=os.path.join(outdir, "checkpoint_it{1}_fold{0}.pt".format(index, it)))

    weight = None#cal_weights()
    criterion = nn.CrossEntropyLoss(weight=weight)
    # criterion = FocalLossWithOutOneHot(gamma=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    avg_train_losses, avg_valid_losses = [], []
    cols = ["loss", "F1", "val_loss", "val_F1", "lr"]
    log_df = pd.DataFrame(columns=cols)
    cnt = 0
    res = 0
    for epoch in range(EPOCHS):
        print('**********************************')
        print("Folder : {} Epoch : {}".format(index, epoch))
        print("Curr learning_rate: {:0.9f}".format(optimizer.param_groups[0]['lr']))
        train_losses, valid_losses = [], []
        tr_loss_cls_item, val_loss_cls_item = [], []

        model.train()  # prep model for training
        train_preds, train_true = torch.Tensor([]).to(device), torch.LongTensor([]).to(device)

        for x, y in train_dataloader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            predictions = model(x)

            if val_score <= 0.7:
                predictions_ = predictions.view(-1, predictions.shape[-1])
            else:
                predictions_ = F.softmax(predictions.view(-1, predictions.shape[-1]))
            y_ = y.view(-1)

            loss = criterion(predictions_, y_)

            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training lossa
            train_losses.append(loss.item())
            train_true = torch.cat([train_true, y_], 0)
            train_preds = torch.cat([train_preds, predictions_], 0)

        model.eval()  # prep model for evaluation
        val_preds, val_true = torch.Tensor([]).to(device), torch.LongTensor([]).to(device)
        print('EVALUATION')
        with torch.no_grad():
            for x, y in valid_dataloader:
                x = x.to(device)
                y = y.to(device)

                predictions = model(x)

                if val_score <= 0.9:
                    predictions_ = predictions.view(-1, predictions.shape[-1])
                else:
                    predictions_ = F.softmax(predictions.view(-1, predictions.shape[-1]))
                    cnt += 1
                y_ = y.view(-1)

                loss = criterion(predictions_, y_)

                valid_losses.append(loss.item())

                val_true = torch.cat([val_true, y_], 0)
                val_preds = torch.cat([val_preds, predictions_], 0)

        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        print("train_loss: {:0.6f}, valid_loss: {:0.6f}".format(train_loss, valid_loss))

        train_score = f1_score(train_true.cpu().detach().numpy(), train_preds.cpu().detach().numpy().argmax(1),
                               labels=list(range(11)), average='macro')

        val_score = f1_score(val_true.cpu().detach().numpy(), val_preds.cpu().detach().numpy().argmax(1),
                             labels=list(range(11)), average='macro')
        if cnt >= 5:
            schedular.step(val_score)
            res = early_stopping(val_score, model)
        print("train_f1: {:0.6f}, valid_f1: {:0.6f}".format(train_score, val_score))

        if res == 2:
            print("Early Stopping")
            print('folder %d global best val max f1 model score %f' % (index, early_stopping.best_score))
            break
        elif res == 1:
            print('save folder %d global val max f1 model score %f' % (index, val_score))
        tmp = pd.DataFrame([[train_loss, train_score, valid_loss, val_score, optimizer.param_groups[0]['lr']]], columns=cols)
        log_df = pd.concat([log_df, tmp], axis=0)
        log_df.to_csv('logs/log_it{1}_{0}.csv'.format(index, it),index=False)
    print('Folder {} finally best global max f1 score is {}'.format(index, early_stopping.best_score))
    oof_score.append(round(early_stopping.best_score, 6))

    model.eval()
    pred_list = []
    with torch.no_grad():
        for x, y in test_dataloader:
            x = x.to(device)
            y = y.to(device)

            predictions = model(x)
            predictions_ = predictions.view(-1, predictions.shape[-1]) # shape [128, 4000, 11]
            #print(predictions.shape, F.softmax(predictions_, dim=1).cpu().numpy().shape)
            pred_list.append(F.softmax(predictions_, dim=1).cpu().numpy()) # shape (512000, 11)
        test_preds = np.vstack(pred_list) # shape [2000000, 11]
        test_preds_all += test_preds

# %%
print('all folder score is:%s'%str(oof_score))
print('OOF mean score is: %f'% (sum(oof_score)/len(oof_score)))
print('Generate submission.............')
submission_csv_path = 'input/sample_submission.csv'
ss = pd.read_csv(submission_csv_path, dtype={'time': str})
test_preds_all = test_preds_all / np.sum(test_preds_all, axis=1)[:, None]
test_pred_frame = pd.DataFrame({'time': ss['time'].astype(str),
                                'open_channels': np.argmax(test_preds_all, axis=1)})
test_pred_frame.to_csv("output/preds_it{}.csv".format(it), index=False)
print('over')
