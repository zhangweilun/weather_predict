import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, conv_k, conv_s):
        super(ConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_k = conv_k
        self.conv_s = conv_s
        # self.b = nn.Parameter(torch.full(size=(in_channels,), fill_value=0.01), requires_grad=True)
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, conv_k, conv_s, bias=True, padding=1)
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0.01)

    def forward(self, x):
        x = self.conv(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, in_channels):
        super(SEBlock, self).__init__()
        self.in_channels = in_channels
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_1 = nn.Linear(self.in_channels, self.in_channels // 16, bias=True)
        self.fc_2 = nn.Linear(self.in_channels // 16, self.in_channels, bias=True)
        self.act_1 = nn.ReLU()
        self.act_2 = nn.Sigmoid()

    def forward(self, x):
        y = self.global_avg_pooling(x)
        y = torch.squeeze(y, dim=-1)
        y = torch.squeeze(y, dim=-1)
        y = self.act_1(self.fc_1(y))
        y = self.act_2(self.fc_2(y))
        y.view(-1, x.shape[1])
        y = torch.unsqueeze(y, dim=-1)
        y = torch.unsqueeze(y, dim=-1)
        # print('se shape: {}'.format((x * y).shape))
        return x * y


class HaarWaveletBlock(nn.Module):
    def __init__(self):
        super(HaarWaveletBlock, self).__init__()
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        feature_map_size = x.shape[1]
        x = torch.squeeze(self.global_avg_pooling(x))
        length = feature_map_size // 2
        temp = torch.reshape(x, (-1, length, 2))
        a = (temp[:, :, 0] + temp[:, :, 1]) / 2
        detail = (temp[:, :, 0] - temp[:, :, 1]) / 2
        length = length // 2
        while length != 16:  # 一级：32，acc：97.5， 二级：16，acc：97.875，三级：8, acc: 98.628, 四级：4，acc: 97.625, 五级：2，acc：97.5，六级：1，acc：97.375
            a = torch.reshape(a, (-1, length, 2))
            detail = torch.cat(((a[:, :, 0] - a[:, :, 1]) / 2, detail), dim=1)
            a = (a[:, :, 0] + a[:, :, 1]) / 2
            length = length // 2
        haar_info = torch.cat((a, detail), dim=1)
        # print('haar shape: {}'.format(haar_info.shape))
        return haar_info


class CAM(nn.Module):

    def __init__(self, in_channels):
        super(CAM, self).__init__()
        self.in_channels = in_channels
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(self.in_channels, self.in_channels // 16, bias=True),
            nn.ReLU(),
            nn.Linear(self.in_channels // 16, self.in_channels, bias=True),
            nn.Sigmoid()
        )
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        y1 = self.avg_pool(x)
        y2 = self.max_pool(x)
        y1, y2 = torch.squeeze(y1), torch.squeeze(y2)
        y1 = self.mlp(y1)
        y2 = self.mlp(y2)
        y = y1 + y2
        y = torch.unsqueeze(y, dim=-1)
        y = torch.unsqueeze(y, dim=-1)
        y = self.sig(y)
        return x * y


class SAM(nn.Module):
    
    def __init__(self):
        super(SAM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, 1, 3, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y1, _ = torch.max(x, dim=1, keepdim=True)
        y2 = torch.mean(x, dim=1, keepdim=True)
        y = torch.cat((y1, y2), dim=1)
        y = self.conv(y)
        return x * y
        
# class BatchNorm(nn.Module):
#     def __init__(self, in_channels, out_channels, train_phase):
#         super(BatchNorm, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         # self.beta = nn.Parameter(torch.full(size=(in_channels,), fill_value=0.0), requires_grad=True)
#         # self.gamma = nn.Parameter(torch.full(size=(in_channels,), fill_value=1.0), requires_grad=True)
#         # self.ema = EMA(0.5)
#         # self.train_phase = train_phase
#
#     def forward(self, x):
#         # mean, var = 0.0, 0.0
#         batch_mean = torch.mean(x, dim=(0, 2, 3))
#         batch_var = torch.var(x, dim=(0, 2, 3))
#         if self.train_phase:
#             self.ema.register('mean', batch_mean)
#             self.ema.register('var', batch_var)
#             self.ema.update('mean', batch_mean)
#             self.ema.update('var', batch_var)
#             mean, var = self.ema.shadow['mean'], self.ema.shadow['var']
#         else:
#             mean, var = torch.mean(batch_mean), torch.mean(var)
#         normed = nn.BatchNorm2d(self.in_channels)




# class EMA():
#     def __init__(self, decay):
#         self.decay = decay
#         self.shadow = {}
#
#     def register(self, name, val):
#         self.shadow[name] = val.clone()
#
#     def get(self, name):
#         return self.shadow[name]
#
#     def update(self, name, x):
#         assert name in self.shadow
#         new_average = (1.0 - self.decay) * x + self.decay * self.shadow[name]
#         self.shadow[name] = new_average.clone()
