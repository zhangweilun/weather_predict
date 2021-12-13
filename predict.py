from visdom import Visdom
from torch import nn
import model
import dataset.dataset as dataset
import util.constant as constant
import torch
import json
import numpy as np
EPSILON = 1e-10
MODEL_PATH = r'.\trained_model\best_model_basic_CBAM_31_mat.pth'


def test():
    net = model.LstmWeather(12, 3, 2)
    # net = SEResNet(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=4)
    # net = nn.DataParallel(net)
    # net = net.cuda()
    net.load_state_dict(torch.load(MODEL_PATH))
    net.eval()
    data_path = constant.DATA_PATH
    test_loss =  []
    criterion =  nn.MSELoss(reduction='mean')
    train_loader, valid_loader, test_loader = dataset.get_data_loader(data_path)
    valid_labels = []
    valid_out = []
    for i, data in enumerate(test_loader):
        images, labels = data
        valid_labels.append(labels.numpy())
        # images, labels = images.cuda(), labels.cuda()
        outputs = net(images)
        loss = criterion(outputs + EPSILON, labels)
        print('Test Loss: {:.6f}'.format(loss))
        valid_out.append(outputs.detach().numpy())
        test_loss.append(loss)
    return valid_labels, valid_out


if __name__ == '__main__':
    vis = Visdom()
    labels, out = test()
    a = []
    b = []
    for item in labels:
        temp = item.tolist()
        for key in temp:
            a.append(key)
    for item in out:
        temp = item.tolist()
        for key in temp:
            b.append(key)
    length = len(a)
    loss_x = range(length)
    vis.line(
        X=np.array(loss_x),
        Y=np.column_stack((np.array(a), np.array(b))),
        opts=dict(showlegend=True, title='predict image',legend = ['real','predict']),
        win='predict image'
        )
