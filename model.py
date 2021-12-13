import torch.nn as nn
import torch


class LstmWeather(nn.Module):

    def __init__(self, input_size, hidden_dim, num_layers):
        super(LstmWeather, self).__init__()
        # Parameters
        self.output_dim = 128
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, dropout=0.1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, self.output_dim)
        # self.linear_1 = nn.Linear(12, 128)
        self.linear_2 = nn.Linear(128, 1)

    def forward(self, x):
        h_t = torch.zeros(2, x.size(0), 3 ,dtype=torch.float32)
        c_t = torch.zeros(2, x.size(0), 3, dtype=torch.float32)
        x = self.lstm(x, (h_t, c_t))
        x = x[0][:, -1, :]
        x = self.fc(x)
        # x = self.sigmoid_1(x)
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
