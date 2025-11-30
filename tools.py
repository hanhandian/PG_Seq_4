from scipy import signal
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
import warnings

warnings.filterwarnings("ignore")


def R(y_obs, y_sim):
    y_obs_mean = y_obs.mean()
    sum1 = 0
    sum2 = 0
    for i in range(len(y_obs)):
        sum1 = sum1 + ((y_sim[i] - y_obs_mean) ** 2)
        sum2 = sum2 + ((y_obs[i] - y_obs_mean) ** 2)
    R = (sum1 / sum2)
    return R


def NSE(y_obs, y_sim):
    y_obs_mean = y_obs.mean()
    sum1 = 0
    sum2 = 0
    for i in range(len(y_obs)):
        sum1 = sum1 + ((y_obs[i] - y_sim[i]) ** 2)
        sum2 = sum2 + ((y_obs[i] - y_obs_mean) ** 2)
    NSE = 1 - (sum1 / sum2)
    return NSE


def PBIAS(y_obs, y_sim):
    sum1 = 0
    sum2 = 0
    for i in range(len(y_obs)):
        sum1 = sum1 + (y_sim[i] - y_obs[i])
        sum2 = sum2 + y_obs[i]
    PBIAS = 100 * (sum1 / sum2)
    return PBIAS


def get_Data(data_path):
    data_raw = pd.read_csv(data_path, encoding='gbk')
    num_columns = data_raw.shape[1]
    data = data_raw.iloc[:, 1:num_columns]  # 以三个特征作为数据
    label = data_raw.iloc[:, num_columns - 1:]  # 取最后一个特征作为标签
    print(data.head())
    print(label.head())
    return data, label


# 数据预处理
def normalization(data, label):
    mm_x = MinMaxScaler()  # 导入sklearn的预处理容器
    mm_y = MinMaxScaler()
    data = data.values  # 将pd的系列格式转换为np的数组格式
    label = label.values
    data = mm_x.fit_transform(data)  # 对数据和标签进行归一化等处理
    label = mm_y.fit_transform(label)
    return data, label, mm_y


def guiyi(syn_data, data):
    a = data.min(axis=0)
    x_std = (syn_data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
    return x_std


# 时间向量转换
def split_windows(data, label, seq_length, pre_length):
    x = []
    y = []
    for i in range(len(data) - seq_length - pre_length + 1):  # range的范围需要减去时间步长和预测步长-1 总步数+1 相抵消
        _x = data[i:(i + seq_length), :]  # 前序列长度
        _y = label[i + seq_length:i + seq_length + pre_length, -1]  # 后一天
        x.append(_x)
        y.append(_y)
    x, y = np.array(x), np.array(y)
    print('x.shape,y.shape=\n', x.shape, y.shape)
    return x, y


def split_data_cpu(args, x, y, split_ratio):
    train_size = int(len(y) * split_ratio)
    test_size = len(y) - train_size

    x_data = np.array(x)
    y_data = np.array(y)

    x_train = np.array(x[0:train_size])
    y_train = np.array(y[0:train_size])
    x_train = x_train.reshape(-1, args.input_dim * args.seq_len)
    y_test = np.array(y[train_size:len(y)])
    x_test = np.array(x[train_size:len(x)])
    x_test = x_test.reshape(-1, args.input_dim * args.seq_len)

    print('x_data.shape,y_data.shape,x_train.shape,y_train.shape,x_test.shape,y_test.shape:\n{}{}{}{}{}{}'
          .format(x_data.shape, y_data.shape, x_train.shape, y_train.shape, x_test.shape, y_test.shape))

    return x_data, y_data, x_train, y_train, x_test, y_test


# 数据分离
def split_data(x, y, split_ratio, device):
    train_size = int(len(y) * split_ratio)
    test_size = len(y) - train_size

    x_data = Variable(torch.Tensor(np.array(x))).to(device)
    y_data = Variable(torch.Tensor(np.array(y))).to(device)

    x_train = Variable(torch.Tensor(np.array(x[0:train_size]))).to(device)
    y_train = Variable(torch.Tensor(np.array(y[0:train_size]))).to(device)
    y_test = Variable(torch.Tensor(np.array(y[train_size:len(y)]))).to(device)
    x_test = Variable(torch.Tensor(np.array(x[train_size:len(x)]))).to(device)

    print('x_data.shape,y_data.shape,x_train.shape,y_train.shape,x_test.shape,y_test.shape:\n{}{}{}{}{}{}'
          .format(x_data.shape, y_data.shape, x_train.shape, y_train.shape, x_test.shape, y_test.shape))

    return x_data, y_data, x_train, y_train, x_test, y_test


def data_generator(x_train, y_train, x_valid, y_valid, batch_size):
    valid_dataset = Data.TensorDataset(x_valid, y_valid)
    train_dataset = Data.TensorDataset(x_train, y_train)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False,
                                               drop_last=True)  # 加载数据集,使数据集可迭代
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                               drop_last=True)  # 加载数据集,使数据集可迭代
    return valid_loader, train_loader


def data_generator_PG(x_train, y_train, batch_size):
    train_dataset = Data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                               drop_last=True)  # 加载数据集,使数据集可迭代
    return train_loader


def direct_result(net, mm_y, pre_len, x_data, y_data):
    y_pred = net.predict(x_data)
    y_pred = mm_y.inverse_transform(y_pred)
    y_data = y_data.cpu()
    # y_test = np.reshape(y_test, (-1, 1))
    y_data = mm_y.inverse_transform(y_data)
    for i in range(pre_len):
        pred = y_pred[:, i]  # 取了序列的第一条来画图
        real = y_data[:, i]  # 取了序列的第一条来画图

        plt.rcParams['font.sans-serif'] = ['FangSong']
        plt.plot(real)
        plt.plot(pred)
        plt.title("第{}天预见期".format(i + 1), fontsize=15, loc='center', color='green')
        plt.legend(('real', 'predict'), fontsize='15')
        plt.show()
        print(len(pred))
        print('第{}天预见期的MAE/RMSE/R2/NSE/PBIAS分别为'.format(i + 1))
        print(mean_absolute_error(real, pred))
        print(np.sqrt(mean_squared_error(real, pred)))
        print(metrics.r2_score(real, pred))  # R2
        print(NSE(real, pred))  # NSE
        print(PBIAS(real, pred))  # NSE
        # pd.DataFrame(pred).to_csv('3.csv')


def identify_peaks(Q, distance=14, **kwargs):
    """
    Identify flood peaks based on scipy find_peaks function.
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html

    Parameters
    ----------
    Q: pandas series of streamflow observations.
    distance: minimal horizontal distance in samples between neighboring peaks (default: 14).
    **kwargs: additional arguments with keywords passed to the scipy.signal.find_peaks() call.

    Returns
    ----------
    peak_time: a sequence of flood peaks' occurrence dates
    """
    peaks_index, _ = signal.find_peaks(
        Q,
        # height=400,
        distance=distance,
        prominence=np.quantile(Q, 0.95),
        width=1,
        rel_height=0.5,
        **kwargs
    )

    # peak_time = Q.iloc[peaks_index].index

    print(f"A total of {len(peaks_index)} flood peaks are identified.")

    return peaks_index
