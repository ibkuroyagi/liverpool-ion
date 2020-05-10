# %%
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
from sklearn.metrics import f1_score
import warnings
from torch.utils.data import Dataset, DataLoader
from wavenet import *
from wavenet2 import *
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)
import os
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# %%
# configurations and main hyperparammeters
EPOCHS = 90  # 150
NNBATCHSIZE = 64
GROUP_BATCH_SIZE = 4000
SEED = 42
LR = 0.001
SPLITS = 5
it = 0
outdir = 'wavenet_models'
print("it:",it)

if not os.path.exists(outdir):
    os.makedirs(outdir)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # tf.random.set_seed(seed)


# %%
# read data
def read_data():
    train = pd.read_csv('input/data-without-drift/train_clean.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})
    test  = pd.read_csv('input/data-without-drift/test_clean.csv', dtype={'time': np.float32, 'signal': np.float32})
    #from https://www.kaggle.com/sggpls/wavenet-with-shifted-rfc-proba and
    # https://www.kaggle.com/c/liverpool-ion-switching/discussion/144645
    # Y_train_proba = np.load("input/ion-shifted-rfc-proba/Y_train_proba.npy")
    # Y_test_proba = np.load("input/ion-shifted-rfc-proba/Y_test_proba.npy")
    
    # for i in range(11):
    #     train[f"proba_{i}"] = Y_train_proba[:, i]
    #     test[f"proba_{i}"] = Y_test_proba[:, i]
        
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
    train['signal'] = (train.signal - train_input_mean) / train_input_sigma
    test['signal'] = (test.signal - train_input_mean) / train_input_sigma
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
    return df

# fillna with the mean and select features for training
def feature_selection(train, test):
    features = [col for col in train.columns if col not in ['index', 'group', 'open_channels', 'time']]
    train = train.replace([np.inf, -np.inf], np.nan)
    test = test.replace([np.inf, -np.inf], np.nan)
    for feature in features:
        feature_mean = pd.concat([train[feature], test[feature]], axis = 0).mean()
        train[feature] = train[feature].fillna(feature_mean)
        test[feature] = test[feature].fillna(feature_mean)
    return train, test, features


def split(GROUP_BATCH_SIZE=4000, SPLITS=5):
    print('Reading Data Started...')
    train, test = read_data()
    train, test = normalize(train, test)
    print('Reading and Normalizing Data Completed')
    print('Creating Features')
    print('Feature Engineering Started...')
    train = run_feat_engineering(train, batch_size=GROUP_BATCH_SIZE)
    test = run_feat_engineering(test, batch_size=GROUP_BATCH_SIZE)
    train, test, features = feature_selection(train, test)
    print(train.head())
    print('Feature Engineering Completed...')

    target = ['open_channels']
    group = train['group']
    kf = GroupKFold(n_splits=SPLITS)
    splits = [x for x in kf.split(train, train[target], group)]
    new_splits = []
    for sp in splits:
        new_split = []
        new_split.append(np.unique(group[sp[0]]))
        new_split.append(np.unique(group[sp[1]]))
        new_split.append(sp[1])
        new_splits.append(new_split)
    target_cols = ['open_channels']
    print(train.head(), train.shape)
    train_tr = np.array(list(train.groupby('group').apply(lambda x: x[target_cols].values))).astype(np.float32)
    train = np.array(list(train.groupby('group').apply(lambda x: x[features].values)))
    test = np.array(list(test.groupby('group').apply(lambda x: x[features].values)))
    print(train.shape, test.shape, train_tr.shape)
    return train, test, train_tr, new_splits


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
        labels = self.labels[idx]

        return [data.astype(np.float32), labels.astype(int)]


# %%
train, test, train_tr, new_splits = split()


# %%
model = Classifier()


# %%
print(outdir)


# %%
test_y = np.zeros([int(2000000/GROUP_BATCH_SIZE), GROUP_BATCH_SIZE, 1])
test_dataset = IronDataset(test, test_y)
test_dataloader = DataLoader(test_dataset, NNBATCHSIZE, shuffle=False, num_workers=8, pin_memory=True)
test_preds_all = np.zeros((2000000, 11))


oof_score = []
for index, (train_index, val_index, _) in enumerate(new_splits[0:], start=0):
    print("Fold : {}".format(index))
    train_dataset = IronDataset(train[train_index], train_tr[train_index])
    train_dataloader = DataLoader(train_dataset, NNBATCHSIZE, shuffle=True, num_workers=8, pin_memory=True)

    valid_dataset = IronDataset(train[val_index], train_tr[val_index])
    valid_dataloader = DataLoader(valid_dataset, NNBATCHSIZE, shuffle=False, num_workers=4, pin_memory=True)

    
    model_path = os.path.join(outdir, "checkpoint_fold_{}_iter_{}.pt".format(index, it))
    model = Classifier(input_size=train.shape[2])
    if index == 0:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    early_stopping = EarlyStopping(patience=40, is_maximize=True, device=device,
                                   checkpoint_path=model_path)

    weight = None#cal_weights()
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.2)
    avg_train_losses, avg_valid_losses = [], []
    cols = ["loss", "F1", "val_loss", "val_F1", "lr"]
    log_df = pd.DataFrame(columns=cols)

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

            predictions_ = predictions.view(-1, predictions.shape[-1])
            y_ = y.view(-1)

            loss = criterion(predictions_, y_)

            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            
            # schedular.step()
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
                predictions_ = predictions.view(-1, predictions.shape[-1])
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

        schedular.step(val_score)
        print("train_f1: {:0.6f}, valid_f1: {:0.6f}".format(train_score, val_score))
        res = early_stopping(val_score, model)
        # print('fres:', res)
        if  res == 2:
            print("Early Stopping")
            print('folder %d global best val max f1 model score %f' % (index, early_stopping.best_score))
            break
        elif res == 1:
            print('save folder %d global val max f1 model score %f' % (index, val_score))
        tmp = pd.DataFrame([[train_loss, train_score, valid_loss, val_score, optimizer.param_groups[0]['lr']]], columns=cols)
        log_df = pd.concat([log_df, tmp], axis=0)
        log_df.to_csv('logs/log_{}.csv'.format(index),index=False)
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
            # print(predictions.shape, F.softmax(predictions_, dim=1).cpu().numpy().shape)
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
test_pred_frame.to_csv("./gru_preds.csv", index=False)
print('over')