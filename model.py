import torch.nn as nn


class LstmWeather(nn.Module):

    def __init__(self):

        super(LstmWeather, self).__init__()
        # Parameters
        self.feature_dim = 12
        self.hidden_dim = 500
        self.num_layers = 3
        self.output_dim = 128
        self.lstm = nn.LSTM(self.feature_dim, self.hidden_dim, self.num_layers, dropout=0.1, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        # self.linear_1 = nn.Linear(12, 128)
        self.sigmoid_1 = nn.Sigmoid()
        self.linear_2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.lstm(x)
        x = x[0][:, -1, :]
        x = self.fc(x)
        x = self.sigmoid_1(x)
        x = self.linear_2(x)
        return x


class Cnn2D(nn.Module):
    """
    对于股票的2d卷积
    """

    def __init__(self):
        super(Cnn2D, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 3))
        self.conv_2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 1))
        self.pool_1 = nn.MaxPool2d((2, 1))
        self.conv_3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 1))
        self.pool_2 = nn.MaxPool2d((2, 1))
        self.flat_1 = nn.Flatten()
        self.ful_1 = nn.Linear(in_features=104, out_features=2)
        self.softmax_1 = nn.Softmax(dim=1)

    def forward(self, x):
        print(x.shape)
        x = self.conv_1(x)
        print(x.shape)
        x = self.conv_2(x)
        print(x.shape)
        x = self.pool_1(x)
        print(x.shape)
        x = self.conv_3(x)
        print(x.shape)
        # x = x.view(x.size(0), -1)
        x = self.pool_2(x)
        print(x.shape)
        x = self.flat_1(x)
        print(x.shape)
        x = self.ful_1(x)
        print(x.shape)
        x = self.softmax_1(x)
        return x
