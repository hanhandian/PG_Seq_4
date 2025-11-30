import argparse
import os

import torch
import numpy as np
import warnings
import pandas as pd
from ax import optimize
from torch import nn
import torch.optim as optim

from tools import data_generator, data_generator_PG
from utils import import_parameter, Data_load, DataProcessing, Dixdata_load, test, \
    choose_model, EarlyStopping, FlowPeak, train, XAI

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='机器学习模型,整合DBO优化算法')
    # model
    parser.add_argument('--device', type=str, default='cuda', help='选择使用的设备，可选值为"cpu"或"cuda"')
    parser.add_argument('--seed', type=int, default=7777, help='设置随机种子')
    parser.add_argument('-choose_model', choices=['SVR', 'RF', 'XGB', 'LSTM', 'GRU', 'BiGRU', 'SeqToSeq'],
                        default='SeqToSeq', help="选择计算模型")
    parser.add_argument('-choose_predict', choices=['滚动', '直接多步'],
                        default='直接多步', help="选择预测方法, 滚动预测则output_size为1, 直接多步则output_size为多步")
    # data
    parser.add_argument('-data_path', default='data', help="数据地址")
    parser.add_argument('-data_name', choices=['SX_data.csv', 'SX_data_1.csv'],
                        default='SX_data.csv', help="数据名称")
    parser.add_argument('-Area', default=1628, help="km2, 流域面积")

    parser.add_argument('-save_path', default='check', help="保存模型的地址")
    parser.add_argument('-model_path', default='1128_check_network', help="模型地址")

    # ON or OFF
    parser.add_argument('-train', type=bool, default=True)
    parser.add_argument('-load', type=bool, default=False, help="导入已训练好的模型(模型参数要与保存模型一致),同时训练模式改为False")
    parser.add_argument('-predict', type=bool, default=False)

    # para
    parser.add_argument('--input_dim', help='input state dimensions', default=3, type=int)
    parser.add_argument('--seq_len', default=5, type=int, help='序列长')
    parser.add_argument('--pre_len', default=1, type=int, help='预测序列长')
    parser.add_argument('--output_size', help='output state dimensions', default=1, type=int)
    parser.add_argument('--seq_output_size', default=1, type=int, help='seqtoseq输出')
    parser.add_argument('--seq_pre_len', default=1, type=int, help='seqtoseq预测')
    parser.add_argument('--split_ratio', default=0.8, type=int, help='训练、验证集划分比例')

    # GRU参数 or SeqToSeq参数
    parser.add_argument('--hidden_dim', help='hidden state dimensions', default=103, type=int)
    parser.add_argument('--num_layers', help='number of layers', default=1, type=int)
    parser.add_argument('--n_dropout', help='dropout', default=0.4, type=float)
    parser.add_argument('--batch_size', help='mini-batch ', default=49, type=int)
    parser.add_argument('--learn_rate', help='learn_rate ', default=0.001, type=float)
    parser.add_argument('--max_epochs', help='max_epochs ', default=500, type=int)

    args = parser.parse_args()

    # 环境
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    # parameter
    params = import_parameter(args)

    # model
    model = choose_model(args, device, params)

    # data-process
    data_raw, label_raw, data111, label111, mm_y111 = Data_load(args)
    # x_data, y_data, x_train, y_train, x_test, y_test = DataProcessing(args, data, label, device)
    # print('{}模型的数据集（{}）加载完成'.format(args.choose_model, args.data_name))

    # 合成数据
    data, label, syn_rain_x, syn_rain_y,  syn_rainless_x, syn_rainless_y, mm_y = Dixdata_load(args)
    # 原始数据集划分
    x_data, y_data, x_train, y_train, x_test, y_test = DataProcessing(args, data, label, device)
    # 暴雨训练数据集
    syn_rain_x_train = torch.Tensor(np.array(syn_rain_x)).to(device)
    syn_rain_y_train = torch.Tensor(np.array(syn_rain_y)).to(device)

    # 无雨训练数据集
    syn_rainless_x_train = torch.Tensor(np.array(syn_rainless_x)).to(device)
    syn_rainless_y_train = torch.Tensor(np.array(syn_rainless_y)).to(device)

    print('物理约束训练数据集加载完成')

    if args.load:
        args.train = False
        model_path = args.save_path + '/' + args.model_path
        model_path = model_path + '.pth'
        model.load_state_dict(torch.load(model_path))
        print('{}模型已加载'.format(model_path))
        _, df = test(args, model, x_test, y_test, mm_y, device)
        # df.to_excel('data/{}-test—data.xlsx'.format(args.model_path), index=False)

        # 流量峰值
        peak_dates = FlowPeak(label_raw, distance=7)
        # 进行可解释性分析
        XAI_data = XAI(args, model, x_train, y_train, data_peakflow)

    if args.train:
        model = train(args, model, x_train, y_train, x_test, y_test, syn_rain_x_train, syn_rain_y_train,
                      syn_rainless_x_train, syn_rainless_y_train)

        path = os.path.join(args.save_path, args.model_path)
        path = path + '.pth'
        torch.save(model.state_dict(), path)

        # test  以后一天的结果为测试
        _, df = test(args, model, x_test, y_test, mm_y, device)
        # _, df2 = test(args, model, x_train, y_train, mm_y, device)
        # _, df3 = test(args, model, x_data, y_data, mm_y, device)
        # df.to_excel('test—data-seq5.xlsx', index=False)
