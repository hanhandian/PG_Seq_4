import os
import random
# import innvestigate
# import shap
import shap
from captum.attr import GradientShap, IntegratedGradients, FeatureAblation
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from xgboost import XGBRegressor
import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tools import get_Data, normalization, split_windows, split_data, split_data_cpu, NSE, PBIAS, data_generator, \
    R, identify_peaks, guiyi
from tools import data_generator, data_generator_PG
# from captum.attr import LayerIntegratedGradients, LayerGradientShap, IntegratedGradients, \
#     GradientShap, FeatureAblation, LayerConductance, KernelShap
from MLmodel import GRU, EncoderDecoderWrapper, LSTM

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为宋体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, save_path, patience=7, verbose=False, delta=0):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, 'best_network.pth')
        torch.save(model.state_dict(), path)  # 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss


def import_parameter(args):
    if args.choose_model in ['SeqToSeq']:
        params = {'input_size': args.input_dim,
                  'hidden_size': args.hidden_dim,
                  'output_size': args.seq_output_size,
                  'num_layers': args.num_layers,
                  'pre_len': args.seq_pre_len,
                  'seq_len': args.seq_len
                  }
    elif args.choose_model in ['GRU']:
        params = {'input_size': args.input_dim,
                  'hidden_size': args.hidden_dim,
                  'output_size': args.output_size,
                  'num_layers': args.num_layers,
                  'n_dropout': args.n_dropout,
                  'device': args.device,
                  }
    elif args.choose_model in ['XGB']:
        params = {
            'learning_rate': args.XGB_learning_rate,
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'min_child_weight': args.min_child_weight,
            # 'subsample': args.subsample,
            # 'colsample_bytree': args.colsample_bytree,
            'gamma': args.XGB_gamma,
            'reg_alpha': args.reg_alpha,
            # 'reg_lambda': args.reg_lambda,
        }
    return params


def choose_model(args, device, params):
    if args.choose_model in ['XGB']:
         model = XGBRegressor(**params)
    elif args.choose_model in ['GRU']:
        model = GRU(**params).to(device)
    elif args.choose_model in ['SeqToSeq']:
        model = EncoderDecoderWrapper(**params).to(device)
    print('{}模型参数导入完成'.format(args.choose_model))
    return model


def Data_load(args):
    data_path = args.data_path + '/' + args.data_name
    data_raw, label_raw = get_Data(data_path)  # 提取数据（去除时间列后，最后一列为label）
    data, label, mm_y = normalization(data_raw, label_raw)  # 归一化
    return data_raw, label_raw, data, label, mm_y


def mixdata_split(args, _data, _label):
    seq_len = args.seq_len
    input_dim = _data.shape[1]

    _data = np.array(_data)
    _label = np.array(_label)
    x = _data.reshape(-1, seq_len, input_dim)
    y = _label
    x, y = np.array(x), np.array(y)
    print('x.shape,y.shape=\n', x.shape, y.shape)
    return x, y


def Dixdata_load(args):
    # 原始数据加载
    data_path = args.data_path + '/' + args.data_name
    data_raw = pd.read_csv(data_path, encoding='gbk')
    num_columns = data_raw.shape[1]
    data = data_raw.iloc[:, 1:num_columns]  # 以三个特征作为数据
    label = data_raw.iloc[:, num_columns - 1:]  # 取最后一个特征作为标签
    data_col = data.columns
    label_col = label.columns

    syn_rain_data = pd.read_pickle('data/syn_rain_data.pkl')
    syn_rain_label = pd.read_pickle('data/syn_rain_label.pkl')

    syn_rainless_data = pd.read_pickle('data/syn_rainless_data.pkl')
    syn_rainless_label = pd.read_pickle('data/syn_rainless_label.pkl')

    data_all = pd.read_pickle('data/data_all.pkl')
    label_all = pd.read_pickle('data/label_all.pkl')

    from sklearn.preprocessing import MinMaxScaler
    # 归一
    mm_x = MinMaxScaler()  # 导入sklearn的预处理容器
    mm_y = MinMaxScaler()
    data_scl = mm_x.fit_transform(data_all)  # 对数据和标签进行归一化等处理
    label_scl = mm_y.fit_transform(label_all)

    # 原始数据进行归一
    data = guiyi(data, data_all)
    syn_rain_data = guiyi(syn_rain_data, data_all)
    syn_rainless_data = guiyi(syn_rainless_data, data_all)

    label = guiyi(label, label_all)
    syn_rain_label = guiyi(syn_rain_label, label_all)
    syn_rainless_label = guiyi(syn_rainless_label, label_all)

    # 训练集切分
    syn_rain_x, syn_rain_y = mixdata_split(args, syn_rain_data, syn_rain_label)
    syn_rainless_x, syn_rainless_y = mixdata_split(args, syn_rainless_data, syn_rainless_label)

    return data.values, label.values, syn_rain_x, syn_rain_y, syn_rainless_x, syn_rainless_y, \
           mm_y


def DataProcessing(args, data, label, device):
    x, y = split_windows(data, label, args.seq_len, args.pre_len)  # 时间窗口的划分
    x_data, y_data, x_train, y_train, x_test, y_test = split_data(x, y, args.split_ratio, device)
    return x_data, y_data, x_train, y_train, x_test, y_test


def train(args, model, x_train, y_train, x_test, y_test, syn_rain_x_train, syn_rain_y_train, syn_rainless_x_train, syn_rainless_y_train):
    print('训练验证功能启用')
    # train
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learn_rate)

    test_loader, train_loader = data_generator(x_train, y_train, x_test, y_test, args.batch_size)
    early_stopping = EarlyStopping(args.save_path, 30)

    syn_rain_loader = data_generator_PG(syn_rain_x_train, syn_rain_y_train, args.batch_size)
    syn_rainless_loader = data_generator_PG(syn_rainless_x_train, syn_rainless_y_train, args.batch_size)

    for epoch in range(args.max_epochs):
        model.train()
        # 原始数据预测训练
        for (batch_x, batch_y), (batch_rain_x, batch_rain_y), (batch_rainless_x, batch_rainless_y), \
                in zip(train_loader, syn_rain_loader, syn_rainless_loader
                       ):
            # 前向传播
            rain_outputs = model(batch_rain_x).to(args.device)
            rain_loss = criterion(rain_outputs, batch_rain_y)

            rainless_outputs = model(batch_rainless_x).to(args.device)
            rainless_loss = criterion(rainless_outputs, batch_rainless_y)

            outputs = model(batch_x).to(args.device)
            loss = criterion(outputs, batch_y)
            # + 0.1 * rainless_loss + 0.1 * fade_loss
            # total_mse = 0.7 * loss + 0.1 * rain_loss + 0.1 * fade_loss + 0.1 * rainless_loss
            total_mse = 0.8 * loss + 0.1 * rain_loss + 0.1 * rainless_loss
            # 反向传播
            optimizer.zero_grad()
            total_mse.backward()
            optimizer.step()

        # total_mse = total_mse_eva + total_mse_prc + loss
        if (epoch + 1) % 10 == 0:
            print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch + 1, args.max_epochs, loss.item()))
            print('总损失为：: [{}/{}], Loss:{:.5f}'.format(epoch + 1, args.max_epochs, total_mse.item()))

        if total_mse < 0.00002:
            break
        # eval_loss = 0
        # model.eval()
        # for i, (batch_x, batch_y) in enumerate(test_loader):
        #     # batch_x = batch_x.permute((1, 0, 2))
        #     outputs = model(batch_x).to(args.device)
        #     loss = criterion(outputs, batch_y)
        #     # 记录误差
        #     eval_loss += loss.item()
        #
        # early_stopping(eval_loss, model)
        # # 达到早停止条件时，early_stop会被置为True
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break  # 跳出迭代，结束训练
    return model


def test(args, model, x_data, y_data, mm_y, device):
    model.eval()
    y_pred = model(x_data).to(device)
    y_pred = y_pred.data.cpu().numpy()
    y_pred = y_pred[:, 0]
    y_pred = np.reshape(y_pred, (-1, 1))

    y_data = y_data.data.cpu().numpy()
    y_data = y_data[:, 0]
    y_data = np.reshape(y_data, (-1, 1))

    y_pred = mm_y.inverse_transform(y_pred)
    y_data = mm_y.inverse_transform(y_data)

    # print(y_pred[335])
    plt.plot(y_data)
    plt.plot(y_pred)
    plt.legend(('real', 'predict'), fontsize='15')
    plt.show()

    print('MAE/RMSE')
    print(mean_absolute_error(y_data, y_pred))
    print(np.sqrt(mean_squared_error(y_data, y_pred)))
    # print("R = ", R(y_data, y_pred))
    print("R2 = ", metrics.r2_score(y_data, y_pred))  # R2
    df_out = pd.DataFrame({'real': y_data.squeeze(),
                           'predict': y_pred.squeeze()})

    return metrics.r2_score(y_data, y_pred), df_out


# 在验证集中使用滚动预测
def one_roll_window(args, data, label):
    train_size = int(len(data) * args.split_ratio)
    test_size = len(data) - train_size
    # 重新切窗口，从验证集往前推一个预见期
    pred_data = np.array(data[train_size - args.pre_len:len(data)])
    pred_data_y = np.array(label[train_size - args.pre_len:len(data)])
    # pred_data = np.array(data[len(data) - args.pre_len - 730:len(data)])

    x = []
    y = []
    for i in range(len(pred_data) - args.seq_len - args.pre_len):  # range的范围需要减去时间步长和预测步长
        # 时间步长 + 预见期步长
        _x = pred_data[i:(i + args.seq_len + args.pre_len), :]
        _y = pred_data_y[i + args.seq_len:i + args.seq_len + args.pre_len, :]
        x.append(_x)
        y.append(_y)
    x = np.array(x)
    y = np.array(y)
    y = np.transpose(y).squeeze()
    print('pred_data.shape=,real_data.shape=\n', x.shape, y.shape)
    return x, y


def one_roll_one(args, model, pred_data, mm_y):
    result = []
    for j in range(args.pre_len):
        test_data = pred_data[:, j:args.seq_len + j, :]
        if args.choose_model in ['SVR', 'RF', 'XGB']:
            test_data = test_data.reshape(-1, args.input_dim * args.seq_len)
            _result = model.predict(test_data)
            _result = _result.squeeze()
            pred_data[:, args.seq_len + j, -1] = _result
            result.append(_result)
        else:
            model.eval()
            test_data = Variable(torch.Tensor(np.array(test_data))).to(args.device)
            _result = model(test_data).to(args.device)
            _result = _result.squeeze().cpu().detach().numpy()
            pred_data[:, args.seq_len + j, -1] = _result
            result.append(_result)
    result = np.array(result)
    result = mm_y.inverse_transform(result)
    return result


def mul_predict_plot(args, model, x_test, y_test, mm_y):
    model.eval()
    _result = model(x_test).to(args.device)
    _result = _result.data.cpu().numpy()
    _test = y_test.data.cpu().numpy()

    cols = len(_result[0])
    for col in range(cols):
        y_pred = _result[:, col]
        _y_test = _test[:, col]
        y_pred = np.reshape(y_pred, (-1, 1))
        _y_test = np.reshape(_y_test, (-1, 1))
        y_pred = mm_y.inverse_transform(y_pred)
        _y_test = mm_y.inverse_transform(_y_test)

        plt.rcParams['font.sans-serif'] = ['FangSong']
        plt.plot(_y_test)
        plt.plot(y_pred)
        plt.title("第{}天预见期".format(col + 1), fontsize=15, loc='center', color='green')
        plt.legend(('real', 'predict'), fontsize='15')
        plt.show()

        print('第{}天预见期的NSE系数'.format(col + 1), NSE(_y_test, y_pred))  # NSE
        print('第{}天预见期的MAE系数'.format(col + 1), mean_absolute_error(_y_test, y_pred))
        print('第{}天预见期的RMSE系数'.format(col + 1), np.sqrt(mean_squared_error(_y_test, y_pred)))


def plot_indicator(args, y_test, y_pred):
    df1 = pd.DataFrame()
    rand_int = random.sample(range(0, 500), 20)
    for a, ran in enumerate(rand_int):
        pred_batch = y_pred[:, ran]
        real_batch = y_test[:, ran]

        df1.insert(0, '第{}个batch的模拟径流'.format(ran), pred_batch)
        df1.insert(0, '第{}个batch的实测径流'.format(ran), real_batch)

    df2 = pd.DataFrame()
    for i in range(args.pre_len):
        pred = y_pred[i]
        test = y_test[i]

        df2.insert(0, '第{}天模拟径流'.format(i + 1), pred)
        df2.insert(0, '第{}天实测径流'.format(i + 1), test)

        pred = pred.reshape(-1, 1)
        test = test.reshape(-1, 1)

        plt.rcParams['font.sans-serif'] = ['FangSong']
        plt.plot(test)
        plt.plot(pred)
        plt.title("第{}天预见期".format(i + 1), fontsize=15, loc='center', color='green')
        plt.legend(('real', 'predict'), fontsize='15')
        plt.show()

        # print(mean_absolute_error(real, pred))
        # print(np.sqrt(mean_squared_error(real, pred)))
        # print(metrics.r2_score(real, pred))  # R2
        print('第{}天预见期的NSE系数'.format(i + 1), NSE(test, pred))  # NSE
        print('第{}天预见期的MAE系数'.format(i + 1), mean_absolute_error(test, pred))
        print('第{}天预见期的RMSE系数'.format(i + 1), np.sqrt(mean_squared_error(test, pred)))
        # print('第{}天预见期的PBIAS'.format(i + 1), PBIAS(test, pred))  # NSE

    return df1, df2


def FlowPeak(data, distance):
    data = data.squeeze()
    peak_dates = identify_peaks(Q=data, distance=distance)
    # print(peak_dates)
    return peak_dates


def XAI(args, model, data, label, peak_dates):
    if args.choose_model in ['SVR']:
        explainer = shap.KernelExplainer(model, data)
        shap_values = explainer.shap_values(data)
        shap.summary_plot(shap_values, data)
        return None
    elif args.choose_model in ['XGB']:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data)
        shap.summary_plot(shap_values, data)
        return None
    elif args.choose_model in ['GRU', 'SeqToSeq']:
        torch.backends.cudnn.enabled = False
        model.eval()

        # 加载算法
        # ig = IntegratedGradients(model)  # 积分梯度
        gs = GradientShap(model)  # shap那边的期望梯度, 整合过来了
        # fa = FeatureAblation(model)
        # 基线
        median_values = torch.median(data, dim=0)[0]
        # baseline = torch.zeros(data.shape[0], data.shape[1], data.shape[2], device=args.device)

        # 计算各场洪水的期望梯度
        shap_data = []
        for data_pe in peak_dates:
            data_pe = data_pe.unsqueeze(0).to(args.device)  # 二维变三维, 符合深度学习模型
            # baseline = torch.zeros(data_pe.shape[0], data_pe.shape[1], data_pe.shape[2], device=args.device)
            # 计算期望梯度
            gs_attr_test = gs.attribute(data_pe, baselines=median_values.unsqueeze(0), n_samples=20)
            gs_attr_test = gs_attr_test.detach().cpu().numpy()

            # gs_attr_test = gs.attribute(data_try, baselines=baseline_try, target=0, n_samples=20)
            # gs_attr_test = gs_attr_test.detach().cpu().numpy().squeeze(0)
            # data_try = data_try.detach().cpu().numpy().squeeze(0)
            shap_data.append(gs_attr_test.squeeze(0))
        shap_data = np.array(shap_data)

        # 三维to二维
        # ig_attr_test = ig_attr_test.reshape(-1, args.seq_len * args.input_dim)
        # data = data.reshape(-1, args.seq_len * args.input_dim)

        # 全局画shap图
        mean_value = torch.median(data, dim=0)[0].unsqueeze(0)
        mean_value = mean_value.repeat(data.shape[0], 1, 1)
        gs_attr_all = gs.attribute(data, baselines=mean_value, n_samples=20)
        gs_attr_all = gs_attr_all.detach().cpu().numpy()
        gs_attr_all = gs_attr_all.reshape(data.shape[0], -1)

        feature_names = [
            f"{feature}_{time_step}"
            for time_step in ["t-5", "t-4", "t-3", "t-2", "t-1"]
            for feature in ["降雨", "蒸发", "历史径流"]
        ]

        data = data.detach().cpu().numpy()
        data = data.reshape(data.shape[0], -1)

        fig = plt.figure()  # <---- initialize figure `fig`
        shap.summary_plot(gs_attr_all, data, feature_names=feature_names, title='PG-1', )  # 三维变二维
        save_path = 'SHAP/{}-shap.png'.format(args.model_path)
        fig.savefig(save_path, dpi=600)  # <---- save `fig` (not current figure)
        plt.close(fig)  # <---- close `fig`

        return shap_data
    return None
