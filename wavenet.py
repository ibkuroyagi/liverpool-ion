import logging
import sys
import time
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    

def encode_mu_law(x, mu=256):
    """PERFORM MU-LAW ENCODING.
    Args:
        x (ndarray): Audio signal with the range from -1 to 1.
        mu (int): Quantized level.
    Returns:
        ndarray: Quantized audio signal with the range from 0 to mu - 1.
    """
    mu = mu - 1
    fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    return np.floor((fx + 1) / 2 * mu + 0.5).astype(np.int64) - 1


def decode_mu_law(y, mu=256):
    """PERFORM MU-LAW DECODING.
    Args:
        x (ndarray): Quantized audio signal with the range from 0 to mu - 1.
        mu (int): Quantized level.
    Returns:
        ndarray: Audio signal with the range from -1 to 1.
    """
    mu = mu - 1
    y = y + 1
    fx = (y - 0.5) / mu * 2 - 1
    x = np.sign(fx) / mu * ((1 + mu) ** np.abs(fx) - 1)
    return x


def initialize(m):
    """INITILIZE CONV WITH XAVIER.
    Arg:
        m (torch.nn.Module): Pytorch nn module instance.
    """
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.0)

    if isinstance(m, nn.ConvTranspose2d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


class OneHot(nn.Module):
    """CONVERT TO ONE-HOT VECTOR.
    Args:
        depth (int): Dimension of one-hot vector
    """

    def __init__(self, depth):
        super(OneHot, self).__init__()
        self.depth = depth

    def forward(self, x):
        """FORWARD CALCULATION.
        Arg:
            x (LongTensor): Long tensor variable with the shape (B, T).
        Returns:
            Tensor: Float tensor variable with the shape (B, depth, T).
        """
        x = x % self.depth
        x = torch.unsqueeze(x, 2)
        x_onehot = x.new_zeros(x.size(0), x.size(1), self.depth).float()

        return x_onehot.scatter_(2, x, 1)


class CausalConv1d(nn.Module):
    """1D DILATED CAUSAL CONVOLUTION."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True):
        #super(CausalConv1d, self).__init__()
        super(self.__class__, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation, bias=bias)

    def forward(self, x):
        """FORWARD CALCULATION.
        Args:
            x (Tensor): Float tensor variable with the shape  (B, C, T).
        Returns:
            Tensor: Float tensor variable with the shape (B, C, T).
        """
        x = self.conv(x)
        if self.padding != 0:
            x = x[:, :, :-self.padding]
        return x


class UpSampling(nn.Module):
    """UPSAMPLING LAYER WITH DECONVOLUTION.
    Args:
        upsampling_factor (int): Upsampling factor.
    """

    def __init__(self, upsampling_factor, bias=True):
        super(UpSampling, self).__init__()
        self.upsampling_factor = upsampling_factor
        self.bias = bias
        self.conv = nn.ConvTranspose2d(1, 1,
                                       kernel_size=(1, self.upsampling_factor),
                                       stride=(1, self.upsampling_factor),
                                       bias=self.bias)

    def forward(self, x):
        """FORWARD CALCULATION.
        Args:
            x (Tensor): Float tensor variable with the shape (B, C, T).
        Returns:
            Tensor: Float tensor variable with the shape (B, C, T'),
                where T' = T * upsampling_factor.
        """
        x = x.unsqueeze(1)  # B x 1 x C x T
        x = self.conv(x)  # B x 1 x C x T'
        return x.squeeze(1)


class WaveNet(nn.Module):
    """CONDITIONAL WAVENET.
    Args:
        n_quantize (int): Number of quantization.
        n_aux (int): Number of aux feature dimension.
        n_resch (int): Number of filter channels for residual block.
        n_skipch (int): Number of filter channels for skip connection.
        dilation_depth (int): Number of dilation depth (e.g. if set 10, max dilation = 2^(10-1)).
        dilation_repeat (int): Number of dilation repeat.
        kernel_size (int): Filter size of dilated causal convolution.
        upsampling_factor (int): Upsampling factor.
    """

    def __init__(self, n_quantize=256, n_aux=28, n_resch=512, n_skipch=256,
                 dilation_depth=10, dilation_repeat=3, kernel_size=2, upsampling_factor=0):
        super(WaveNet, self).__init__()
        self.n_aux = n_aux
        self.n_quantize = n_quantize
        self.n_resch = n_resch
        self.n_skipch = n_skipch
        self.kernel_size = kernel_size
        self.dilation_depth = dilation_depth
        self.dilation_repeat = dilation_repeat
        self.upsampling_factor = upsampling_factor

        self.dilations = [2 ** i for i in range(self.dilation_depth)] * self.dilation_repeat
        self.receptive_field = (self.kernel_size - 1) * sum(self.dilations) + 1

        # for preprocessing
        self.onehot = OneHot(self.n_quantize)
        self.causal = CausalConv1d(self.n_quantize, self.n_resch, self.kernel_size)
        if self.upsampling_factor > 0:
            self.upsampling = UpSampling(self.upsampling_factor)

        # for residual blocks
        self.dil_sigmoid = nn.ModuleList()
        self.dil_tanh = nn.ModuleList()
        self.aux_1x1_sigmoid = nn.ModuleList()
        self.aux_1x1_tanh = nn.ModuleList()
        self.skip_1x1 = nn.ModuleList()
        self.res_1x1 = nn.ModuleList()
        for d in self.dilations:
            self.dil_sigmoid += [CausalConv1d(self.n_resch, self.n_resch, self.kernel_size, d)]
            self.dil_tanh += [CausalConv1d(self.n_resch, self.n_resch, self.kernel_size, d)]
            self.aux_1x1_sigmoid += [nn.Conv1d(self.n_aux, self.n_resch, 1)]
            self.aux_1x1_tanh += [nn.Conv1d(self.n_aux, self.n_resch, 1)]
            self.skip_1x1 += [nn.Conv1d(self.n_resch, self.n_skipch, 1)]
            self.res_1x1 += [nn.Conv1d(self.n_resch, self.n_resch, 1)]

        # for postprocessing
        self.conv_post_1 = nn.Conv1d(self.n_skipch, self.n_skipch, 1)
        self.conv_post_2 = nn.Conv1d(self.n_skipch, self.n_quantize, 1)
        self.linear = nn.Linear(self.n_quantize, self.n_quantize)

    def forward(self, x, h):
        """FORWARD CALCULATION.
        Args:
            x (Tensor): Long tensor variable with the shape (B, T).
            h (Tensor): Float tensor variable with the shape (B, n_aux, T),
        Returns:
            Tensor: Float tensor variable with the shape (B, T, n_quantize).
        """
        # preprocess
        output = self._preprocess(x)
        if self.upsampling_factor > 0:
            h = self.upsampling(h)

        # residual block
        skip_connections = []
        for l in range(len(self.dilations)):
            output, skip = self._residual_forward(
                output, h, self.dil_sigmoid[l], self.dil_tanh[l],
                self.aux_1x1_sigmoid[l], self.aux_1x1_tanh[l],
                self.skip_1x1[l], self.res_1x1[l])
            skip_connections.append(skip)

        # skip-connection part
        output = sum(skip_connections)
        output = self._postprocess(output)
        output = self.linear(output)
        #output = F.softmax(output, dim=2)
        return output

    def _preprocess(self, x):
        x = self.onehot(x).transpose(1, 2)
        output = self.causal(x)
        return output

    def _postprocess(self, x):
        output = F.relu(x)
        output = self.conv_post_1(output)
        output = F.relu(output)  # B x C x T
        output = self.conv_post_2(output).transpose(1, 2)  # B x T x C
        return output

    def _residual_forward(self, x, h, dil_sigmoid, dil_tanh,
                          aux_1x1_sigmoid, aux_1x1_tanh, skip_1x1, res_1x1):
        output_sigmoid = dil_sigmoid(x)
        output_tanh = dil_tanh(x)
        aux_output_sigmoid = aux_1x1_sigmoid(h)
        aux_output_tanh = aux_1x1_tanh(h)
        output = torch.sigmoid(output_sigmoid + aux_output_sigmoid) * \
            torch.tanh(output_tanh + aux_output_tanh)
        skip = skip_1x1(output)
        output = res_1x1(output)
        output = output + x
        return output, skip
    
# MSE model
class WaveNet0(nn.Module):
    """CONDITIONAL WAVENET.
    Args:
        n_quantize (int): Number of quantization.
        n_feature (int): Number of feature dimension.
        n_resch (int): Number of filter channels for residual block.
        n_skipch (int): Number of filter channels for skip connection.
        dilation_depth (int): Number of dilation depth (e.g. if set 10, max dilation = 2^(10-1)).
        dilation_repeat (int): Number of dilation repeat.
        kernel_size (int): Filter size of dilated causal convolution.
        upsampling_factor (int): Upsampling factor.
    """

    def __init__(self, n_quantize=256, n_resch=512, n_skipch=256, n_feature=10,
                 dilation_depth=10, dilation_repeat=3, kernel_size=2):
        super(self.__class__, self).__init__()
        self.n_quantize = n_quantize
        self.n_feature = n_feature
        self.n_resch = n_resch
        self.n_skipch = n_skipch
        self.kernel_size = kernel_size
        self.dilation_depth = dilation_depth
        self.dilation_repeat = dilation_repeat

        self.dilations = [2 ** i for i in range(self.dilation_depth)] * self.dilation_repeat
        self.receptive_field = (self.kernel_size - 1) * sum(self.dilations) + 1

        # for preprocessing
        self.onehot = OneHot(self.n_quantize)
        self.causal = CausalConv1d(self.n_feature, self.n_resch, self.kernel_size)


        # for residual blocks
        self.dil_sigmoid = nn.ModuleList()
        self.dil_tanh = nn.ModuleList()
        self.skip_1x1 = nn.ModuleList()
        self.res_1x1 = nn.ModuleList()
        for d in self.dilations:
            self.dil_sigmoid += [CausalConv1d(self.n_resch, self.n_resch, self.kernel_size, d)]
            self.dil_tanh += [CausalConv1d(self.n_resch, self.n_resch, self.kernel_size, d)]
            self.skip_1x1 += [nn.Conv1d(self.n_resch, self.n_skipch, 1)]
            self.res_1x1 += [nn.Conv1d(self.n_resch, self.n_resch, 1)]

        # for postprocessing
        self.conv_post_1 = nn.Conv1d(self.n_skipch, self.n_skipch, 1)
        self.conv_post_2 = nn.Conv1d(self.n_skipch, 32, 1)
        self.linear = nn.Linear(32, 1)

    def forward(self, x):
        """FORWARD CALCULATION.
        Args:
            x (Tensor): Long tensor variable with the shape (B, n, T).
        Returns:
            Tensor: Float tensor variable with the shape (B, T).
        """
        # preprocess
        output = self.causal(x)

        # residual block
        skip_connections = []
        for l in range(len(self.dilations)):
            output, skip = self._residual_forward(
                output, self.dil_sigmoid[l], self.dil_tanh[l],
                self.skip_1x1[l], self.res_1x1[l])
            skip_connections.append(skip)

        # skip-connection part
        output = sum(skip_connections)
        output = self._postprocess(output)
        output = self.linear(output)
        return output.reshape(-1,)



    def _postprocess(self, x):
        output = F.relu(x)
        output = self.conv_post_1(output)
        output = F.relu(output)  # B x C x T
        output = self.conv_post_2(output).transpose(1, 2)  # B x T x C
        return output

    def _residual_forward(self, x, dil_sigmoid, dil_tanh, skip_1x1, res_1x1):
        output_sigmoid = dil_sigmoid(x)
        output_tanh = dil_tanh(x)
        output = torch.sigmoid(output_sigmoid) * torch.tanh(output_tanh)
        skip = skip_1x1(output)
        output = res_1x1(output)
        output = output + x
        return output, skip

# CrossEntropy model
class WaveNet_CE0(nn.Module):
    """CONDITIONAL WAVENET.
    Args:
        n_quantize (int): Number of quantization.
        n_feature (int): Number of feature dimension.
        n_resch (int): Number of filter channels for residual block.
        n_skipch (int): Number of filter channels for skip connection.
        dilation_depth (int): Number of dilation depth (e.g. if set 10, max dilation = 2^(10-1)).
        dilation_repeat (int): Number of dilation repeat.
        kernel_size (int): Filter size of dilated causal convolution.
        upsampling_factor (int): Upsampling factor.
    """
    def __init__(self, n_quantize=256, n_resch=512, n_skipch=256, n_feature=10,
                 dilation_depth=10, dilation_repeat=3, kernel_size=2):
        super(self.__class__, self).__init__()
        self.n_quantize = n_quantize
        self.n_feature = n_feature
        self.n_resch = n_resch
        self.n_skipch = n_skipch
        self.kernel_size = kernel_size
        self.dilation_depth = dilation_depth
        self.dilation_repeat = dilation_repeat

        self.dilations = [2 ** i for i in range(self.dilation_depth)] * self.dilation_repeat
        self.receptive_field = (self.kernel_size - 1) * sum(self.dilations) + 1

        # for preprocessing
        self.causal = CausalConv1d(self.n_feature, self.n_resch, self.kernel_size)


        # for residual blocks
        self.dil_sigmoid = nn.ModuleList()
        self.dil_tanh = nn.ModuleList()
        self.skip_1x1 = nn.ModuleList()
        self.res_1x1 = nn.ModuleList()
        for d in self.dilations:
            self.dil_sigmoid += [CausalConv1d(self.n_resch, self.n_resch, self.kernel_size, d)]
            self.dil_tanh += [CausalConv1d(self.n_resch, self.n_resch, self.kernel_size, d)]
            self.skip_1x1 += [nn.Conv1d(self.n_resch, self.n_skipch, 1)]
            self.res_1x1 += [nn.Conv1d(self.n_resch, self.n_resch, 1)]

        # for postprocessing
        self.conv_post_1 = nn.Conv1d(self.n_skipch, self.n_skipch, 1)
        self.conv_post_2 = nn.Conv1d(self.n_skipch, self.n_quantize, 1)
        self.linear = nn.Linear(self.n_quantize, self.n_quantize)

    def forward(self, x):
        """FORWARD CALCULATION.
        Args:
            x (Tensor): Long tensor variable with the shape (B, n, T).
        Returns:
            Tensor: Float tensor variable with the shape (B, T, n).
        """
        # preprocess
        output = self.causal(x)

        # residual block
        skip_connections = []
        for l in range(len(self.dilations)):
            output, skip = self._residual_forward(
                output, self.dil_sigmoid[l], self.dil_tanh[l],
                self.skip_1x1[l], self.res_1x1[l])
            skip_connections.append(skip)

        # skip-connection part
        output = sum(skip_connections)
        output = self._postprocess(output)
        output = self.linear(output)
        return output

    def _postprocess(self, x):
        output = F.relu(x)
        output = self.conv_post_1(output)
        output = F.relu(output)  # B x C x T
        output = self.conv_post_2(output).transpose(1, 2)  # B x T x C
        return output

    def _residual_forward(self, x, dil_sigmoid, dil_tanh, skip_1x1, res_1x1):
        output_sigmoid = dil_sigmoid(x)
        output_tanh = dil_tanh(x)
        output = torch.sigmoid(output_sigmoid) * torch.tanh(output_tanh)
        skip = skip_1x1(output)
        output = res_1x1(output)
        output = output + x
        return output, skip
    

class WaveNet_conv2d(nn.Module):
    """CONDITIONAL WAVENET.
    Args:
        n_feature (int): Number of feature dimension.
        n_resch (int): Number of filter channels for residual block.
        n_skipch (int): Number of filter channels for skip connection.
        dilation_depth (int): Number of dilation depth (e.g. if set 10, max dilation = 2^(10-1)).
        dilation_repeat (int): Number of dilation repeat.
        kernel_size (int): Filter size of dilated causal convolution.
        upsampling_factor (int): Upsampling factor.
        in_channles (int): Number of in_channel's dimension.
        out_channles (int): Number of out_channel's dimension.
        time_conv_frame (int): Filter size of the first 2D convolution.
    """

    def __init__(self, n_resch=512, n_skipch=256, n_feature=10,
                 dilation_depth=10, dilation_repeat=3, kernel_size=2,
                 in_channels=1, out_channels=32, time_conv_frame=64):
        super(self.__class__, self).__init__()
        self.n_feature = n_feature
        self.n_resch = n_resch
        self.n_skipch = n_skipch
        self.kernel_size = kernel_size
        self.dilation_depth = dilation_depth
        self.dilation_repeat = dilation_repeat
        self.in_channels = in_channels
        self.out_channels = out_channels
        if time_conv_frame % 2 == 0:
            time_conv_frame += 1
        self.time_conv_frame = time_conv_frame

        self.dilations = [2 ** i for i in range(self.dilation_depth)] * self.dilation_repeat
        self.receptive_field = (self.kernel_size - 1) * sum(self.dilations) + 1

        # for preprocessing
        kernel_size = (n_feature, time_conv_frame)
        padding_2d = (0, (time_conv_frame - 1) // 2 )
        self.conv2d = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=kernel_size, padding=padding_2d)
        self.causal = CausalConv1d(self.out_channels, self.n_resch, self.kernel_size)

        # for residual blocks
        self.dil_sigmoid = nn.ModuleList()
        self.dil_tanh = nn.ModuleList()
        self.skip_1x1 = nn.ModuleList()
        self.res_1x1 = nn.ModuleList()
        for d in self.dilations:
            self.dil_sigmoid += [CausalConv1d(self.n_resch, self.n_resch, self.kernel_size, d)]
            self.dil_tanh += [CausalConv1d(self.n_resch, self.n_resch, self.kernel_size, d)]
            self.skip_1x1 += [nn.Conv1d(self.n_resch, self.n_skipch, 1)]
            self.res_1x1 += [nn.Conv1d(self.n_resch, self.n_resch, 1)]

        # for postprocessing
        self.conv_post_1 = nn.Conv1d(self.n_skipch, self.n_skipch, 1)
        self.conv_post_2 = nn.Conv1d(self.n_skipch, 32, 1)
        self.linear = nn.Linear(32, 1)

    def forward(self, x):
        """FORWARD CALCULATION.
        Args:
            x (Tensor): Long tensor variable with the shape (B, in_channels, n_feature, T).
        Returns:
            Tensor: Float tensor variable with the shape (B, T).
        """
        # preprocess
        # x (Tensor): Double tensor variable with the shape (B, in_channels, n_feature, T).
        output = self.conv2d(x).squeeze()
        # print(output.shape)
        # x (Tensor): Double tensor variable with the shape (B, out_channels, T - n_feature)
        output = self.causal(output)

        # residual block
        skip_connections = []
        for l in range(len(self.dilations)):
            output, skip = self._residual_forward(
                output, self.dil_sigmoid[l], self.dil_tanh[l],
                self.skip_1x1[l], self.res_1x1[l])
            skip_connections.append(skip)

        # skip-connection part
        output = sum(skip_connections)
        output = self._postprocess(output)
        output = self.linear(output)
        return output.squeeze()


    def _postprocess(self, x):
        output = F.relu(x)
        output = self.conv_post_1(output)
        output = F.relu(output)  # B x C x T
        output = self.conv_post_2(output).transpose(1, 2)  # B x T x C
        return output

    def _residual_forward(self, x, dil_sigmoid, dil_tanh, skip_1x1, res_1x1):
        output_sigmoid = dil_sigmoid(x)
        output_tanh = dil_tanh(x)
        output = torch.sigmoid(output_sigmoid) * torch.tanh(output_tanh)
        skip = skip_1x1(output)
        output = res_1x1(output)
        output = output + x
        return output, skip

    
    
class EarlyStopping:
    def __init__(self, patience=5, checkpoint_path='checkpoint.pt', device="cpu"):
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.counter, self.best_score = 0, None
        self.device = device
        self.epoch = 0

    def load_best_weights(self, model):
        model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))

    def __call__(self, score, model, mode="min", epoch=0):
        if mode == "max":
            if (self.best_score is None) or (score > self.best_score):
                torch.save(model.to('cpu').state_dict(), self.checkpoint_path)
                model.to(self.device)
                self.best_score, self.counter = score, 0
                self.epoch = epoch
                return 1
            else:
                self.counter += 1
                
                if self.counter >= self.patience:
                    return 2
        if mode == "min":
            if (self.best_score is None) or (score < self.best_score):
                torch.save(model.to('cpu').state_dict(), self.checkpoint_path)
                model.to(self.device)
                self.best_score, self.counter = score, 0
                self.epoch = epoch
                return 1
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    return 2
        return 0

def train_val_test(X, look_back=200, fold=0):
    """
    split data by group
    Paramns
    X : ndarray (time, n_feature)
    look_back : int default 200
        how many time point the model looks back
    fold : int default 0
        the type of spliting data. you must select "fold" in [0, 1, 2, 3].
    """
    X = pd.DataFrame(X)
    X = X.iloc[:(X.shape[0]//look_back)*look_back]
    X["group"] = X.groupby(X.index.astype(int) // look_back, sort=False).agg(['ngroup']).iloc[:, 0].values
    X['group'] = X['group'].astype(np.uint16)
    exclude_cols =["group"]
    feature_cols = [col for col in X.columns if col not in exclude_cols]
    if len(feature_cols) == 1:
        # if you input target data, the shape want to be (batch, look_back)
        # In this line, make data 2 dim
        X = np.array(list(X.groupby('group').apply(lambda x: x[feature_cols].values))).squeeze()
    else:
        # In this line, make data (batch, n_feature, look_back)
        X = np.array(list(X.groupby('group').apply(lambda x: x[feature_cols].values))).transpose(0, 2, 1)
        
    l = len(X)
    if fold == 0:
        X_train = X[:int(l*0.5)]
        X_valid = X[int(l*0.5):int(l*0.75)]
        X_test = X[int(l*0.75):]
    elif fold == 1:
        X_train = X[int(l*0.25):int(l*0.75)]
        X_valid = X[int(l*0.75):]
        X_test = X[:int(l*0.25)]
    elif fold == 2:
        X_train = X[int(l*0.5):]
        X_valid = X[:int(l*0.25)]
        X_test = X[int(l*0.25):int(l*0.5)]
    elif fold == 3:
        X_train = np.concatenate([X[:int(l*0.25)], X[int(l*0.75):]], 0)
        X_valid = X[int(l*0.25):int(l*0.5)]
        X_test = X[int(l*0.5):int(l*0.75)]
    return X_train, X_valid, X_test

class sampleDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        X = self.X[idx]
        y = self.y[idx]
        return [X.astype(np.float32), y.astype(np.float32)]


class conv2dDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        X = self.X[idx][None, ::]
        y = self.y[idx]
        return [X.astype(np.float32), y.astype(np.float32)]
    
    
    
class EMDLoss(nn.Module):
    def __init__(self, mu, eps=1e-3, device="cuda"):
        """
        Usage
        -----------------------------
        mu = 5
        true_y = torch.tensor([4, 2, 1])
        pred_y = torch.tensor([
            [0.1, 0.2, 0.1, 0.1, 0.5],
            [0.3, 0.35, 0.05, 0.2, 0.1],
            [0.5, 0.3, 0.1, 0.1, 0.0]
        ])
        pred_y1 = torch.tensor([
            [0.1, 0.2, 0.1, 0.5, 0.1],
            [0.35, 0.1, 0.05, 0.2, 0.3],
            [0.3, 0.5, 0.1, 0.0, 0.1]
        ])
        pred_y2 = torch.tensor([
            [0.1, 0.2, 0.1, 0.5, 0.1],
            [0.35, 0.30, 0.05, 0.2, 0.1],
            [0.3, 0.5, 0.1, 0.1, 0.0]
        ])
        emd = EMDLoss(mu, eps=0)
        loss = emd(pred_y, true_y)
        loss1 = emd(pred_y1, true_y)
        loss2 = emd(pred_y2, true_y)
        print(loss)
        print(loss1)
        print(loss2)
        ---------------------------------
        tensor(2.2352, dtype=torch.float64)
        tensor(3.6373, dtype=torch.float64)
        tensor(2.8237, dtype=torch.float64)
        """
        super(self.__class__, self).__init__()
        tmp1 = np.arange(mu).reshape(-1, 1)
        tmp2 = np.arange(mu).reshape(1, -1)
        self.mu = mu
        self.device = device
        self.d = (decode_mu_law(tmp1, mu=mu) - decode_mu_law(tmp2, mu=mu)) ** 2
        self.eps = torch.tensor(eps).to(device)

    def forward(self, pred_y, true_y):
        """
        Earth mover's distance loss
        Pramns
        pred_y : torch tensor, torch.float32
            pred_y
        true_y : torch tensor, torch.int64
            true_y
        """
        one_hot_y = torch.eye(self.mu)[true_y].to(self.device)
        p = torch.log(torch.abs(1 - one_hot_y - pred_y))
        d = torch.tensor([self.d[true_y.cpu().numpy()]]).to(self.device) + self.eps
        loss = -torch.sum(d * p)
        return loss