import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
import numpy as np
import pandas as pd
import util.constant as constant
from sklearn.preprocessing import MinMaxScaler


def split_data(data_dir: str, env: constant.Env) -> pd.DataFrame:
    """
    划分数据集
    """
    data = pd.read_csv(data_dir)

    mm = MinMaxScaler()
    train_num = int(len(data) * 0.7)
    train_data = data.iloc[0:train_num, :]
    train_data_x = train_data.iloc[:, 0:12]
    train_data_y = train_data.iloc[:, 12:13]
    train_data_x = mm.fit_transform(train_data_x)
    train_data_x = pd.DataFrame(train_data_x)
    train_data_x.loc[:,"y"] =train_data_y
    valid_num = int(len(data) * 0.2)
    valid_data = data.iloc[train_num:train_num + valid_num, :]
    valid_data_x = valid_data.iloc[:, 0:12]
    valid_data_y = valid_data.iloc[:, 12:13]
    valid_data_x = mm.transform(valid_data_x)
    valid_data_x = pd.DataFrame(valid_data_x)
    valid_data_y = valid_data_y.reset_index()
    valid_data_y = valid_data_y.drop(columns=['index'])
    valid_data_x.loc[:,"y"] =valid_data_y


    mm_test = MinMaxScaler()
    test_data = data.iloc[train_num + valid_num:, :]
    test_data_x = train_data.iloc[:, 0:12]
    test_data_y = train_data.iloc[:, 12:13]
    test_data_x = mm_test.fit_transform(test_data_x)
    test_data_x = pd.DataFrame(test_data_x)
    test_data_x.loc[:, "y"] = test_data_y
    if env.TRAIN == env:
        return train_data_x
    elif env.VALID == env:
        return valid_data_x
    elif env.TEST == env:
        return test_data


def get_data_loader(data_dir: str) -> (DataLoader, DataLoader, DataLoader):
    """
    返回data_loader
    :param data_dir: 训练数据所在的文件路径
    :return: data_loader
    """
    train = split_data(data_dir, constant.Env.TRAIN)
    train_x, train_y = data_handler(train, regression=True, feature_nums=12)
    valid = split_data(data_dir, constant.Env.VALID)
    valid_x, valid_y = data_handler(valid, regression=True, feature_nums=12)
    test = split_data(data_dir, constant.Env.TEST)
    test_x, test_y = data_handler(test, regression=True, feature_nums=12)
    train_x = torch.from_numpy(train_x.astype(np.float32))
    # train_x = torch.from_numpy(train_x.astype(np.float32)).unsqueeze(1)
    valid_x = torch.from_numpy(valid_x.astype(np.float32))
    test_x = torch.from_numpy(test_x.astype(np.float32))
    train_y = torch.from_numpy(train_y.astype(np.float32))
    valid_y = torch.from_numpy(valid_y.astype(np.float32))
    test_y = torch.from_numpy(test_y.astype(np.float32))
    train_dataset = SpDataset(train_x, train_y)
    valid_dataset = SpDataset(valid_x, valid_y)
    test_dataset = SpDataset(test_x, test_y)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    return train_loader, valid_loader, test_loader


def data_handler(data: pd.DataFrame, windows_size=5, feature_nums=3, regression=True) -> (np.array, np.array):
    """
        根据给定的序列data，生成数据集
        数据集分为输入和输出，每一个输入的长度为days_for_train，每一个输出的长度为1。
        也就是说用days_for_train天的数据，对应下一天的数据。
        feature_nums:输入的特征数量 默认最后一列为y,并且列名为y
        若给定序列的长度为d，将输出长度为(d-days_for_train+1)个输入/输出对
    """
    # data.drop('trade_date', axis=1, inplace=True)
    dataset_x, dataset_y = [], []
    if regression:
        for i in range(len(data) - windows_size):
            x = data.iloc[i:i + windows_size, :feature_nums]
            dataset_x.append(x)
        dataset_y = data.iloc[windows_size:, data.shape[1] - 1:data.shape[1]]
        return np.array(dataset_x), np.array(dataset_y)
    else:
        for i in range(len(data) - windows_size):
            x = data.iloc[i:i + windows_size, :feature_nums]
            dataset_x.append(x)
        y = data.iloc[windows_size - 1:, data.shape[1] - 1:data.shape[1]]
        # 相邻两行相减
        y["tump"] = y["y"].shift(1)
        gap_y = y["y"] - y["tump"]
        for i, v in gap_y.items():
            # print(i, v)
            if pd.isna(v):
                continue
            if v >= 0:
                dataset_y.append(1)
            else:
                dataset_y.append(0)
        return np.array(dataset_x), np.array(dataset_y)


class SpDataset(Dataset):
    """
      TensorDataset继承Dataset, 重载了__init__(), __getitem__(), __len__()
      实现将一组Tensor数据对封装成Tensor数据集
      能够通过index得到数据集的数据，能够通过len，得到数据集大小
      """

    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        # return self.data_tensor.size(0)
        return self.data_tensor.shape[0]
