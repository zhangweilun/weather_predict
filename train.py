import json

import numpy as np
import pandas as pd
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import util.constant as constant
import model
import dataset.dataset as dataset
import warnings
from visdom import Visdom
from torch import Tensor

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # 获取绘图对象，相当于plt
    vis = Visdom()

    gpu = torch.device('cuda')
    epochs = constant.EPOCHS
    learning_rate = constant.LEARNING_RATE
    weight_decay = constant.WEIGHT_DECAY
    epsilon = constant.EPSILON
    max_acc = constant.MAX_ACC
    data_path = constant.DATA_PATH
    model = model.LstmWeather()
    train = dataset.split_data(data_path, env=constant.Env.TRAIN)
    valid = dataset.split_data(data_path, env=constant.Env.VALID)
    train_loader, valid_loader, test_loader = dataset.get_data_loader(data_path)
    # train_datum, valid_datum, test_datum = dataset.get_dataset(r"D:\project\zwl\sp500\data")
    # train_loader, valid_loader, test_loader = dataset.get_data_loader(r"D:\project\zwl\sp500\data")
    train_loss = []
    valid_loss = []
    criterion = nn.CrossEntropyLoss()
    # 计算损失函数（均方误差)
    cost = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.cuda()

    # print("模型现在位于:{}".format(model.device))
    for epoch in range(epochs):
        print('*' * 30, 'epoch {}'.format(epoch + 1), '*' * 30)
        model.train()
        loss = 0.0
        for i, data in enumerate(train_loader):
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            # print("imgaes现在位于:{},labels现在位于:{}".format(images.device, labels.device))
            optimizer.zero_grad()
            outputs = model(images)
            loss = cost(outputs + epsilon, labels)
            loss.backward()
            optimizer.step()
        print('Finish {} epoch\nLoss: {:.6f},'.format(epoch + 1, loss))
        train_loss.append(loss)

        # 验证
        model.eval()
        eval_loss = 0.0
        for data in valid_loader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            loss = cost(outputs, labels)
            eval_loss = eval_loss + loss.item()
        print('Valid Loss: {:.6f}'.format(eval_loss * 100 / len(valid)))
        valid_loss.append(eval_loss)
        if eval_loss < 100:
            torch.save(model.state_dict(), '.\\trained_model\\best_model_basic_CBAM_31_mat.pth')
    print(valid_loss)
    train_loss = torch.tensor(train_loss, device='cpu')
    length = len(train_loss)
    loss_x = range(length)
    vis.line(
        X=list(loss_x),  # x坐标
        Y=list(train_loss),  # y值
        win="Loss",  # 窗口id
        name="Train Loss",  # 线条名称
        update=None,  # 已添加方式加入
        opts={
            # 显示网格   # x轴标签  # y轴标签
            'showlegend': True, 'title': "Loss", 'xlabel': "x1", 'ylabel': "y1", }, )