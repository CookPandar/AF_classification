import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F
import math
import numpy as np
import random
import torchvision.models as torch_models
num_ftrs=256

#ResNet-50  代替 AlexNetforEcg_DS1_to_DS2(nn.Module)
class ResNetforEcg_DS1_to_DS2(nn.Module):
    '''input tensor size:(None,3,m,n)   m*n的输入图片'''
    def __init__(self):
        super(ResNetforEcg_DS1_to_DS2, self).__init__()
        resnet_pretrained = torch_models.resnet50(pretrained=True)
        num_ftrs = resnet_pretrained.fc.in_features
        self.fc = nn.Sequential(
            nn.Linear(num_ftrs * 15, num_ftrs* 10),
            # nn.Linear(256 * 15, 256 * 5),
            # nn.Linear(256 * 14, 256 * 10),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),#0.3
            nn.BatchNorm1d(num_ftrs * 10),
        )
        self.features = nn.Sequential(resnet_pretrained,self.fc)


    def forward(self, x):
        x = self.features(x)
        # print("feature size:", x.size())
        # x = x.view(x.size(0), self.num_flat_features(x))
        # print(x.size())
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



class EcgClassifier(nn.Module):
    """Feature classifier class for MNIST -> MNIST-M experiment in ATDA."""

    def __init__(self, dropout_keep=None, num_classes=5):
        """Init classifier."""
        super(EcgClassifier, self).__init__()

        self.dropout_keep = dropout_keep

        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs * 10, num_ftrs * 5),
            # nn.Linear(256 *5, 256 * 1),
            # nn.BatchNorm1d(256 * 5),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_keep),  # 0.5
            nn.Linear(num_ftrs* 5, num_classes),

        )


    def forward(self, x):
        """Forward classifier."""
        out = self.classifier(x)
        return out


class AlexNetforEcg_DS1_to_DS2(nn.Module):
    '''input tensor size:(None,1,1,251)'''
    def __init__(self):
        super(AlexNetforEcg_DS1_to_DS2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,64, kernel_size=(1, 5), padding=(0, 0)),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),#（N,64,1,62)


            nn.Conv2d(64, 192, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),#(N,192,1,30)



            nn.Conv2d(192, 256, kernel_size=(1, 5), padding=(0, 2)),#(N,_,1,30)
            # nn.Conv2d(192, 256, kernel_size=(1, 3), padding=(0, 1)),
            # # nn.BatchNorm2d(384),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(inplace=True),
            # nn.Conv2d(384, 384, kernel_size=(1, 1), padding=(0, 0)),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(384, 256, kernel_size=(1, 1), padding=(0, 0)),
            # nn.ReLU(inplace=True),

            # nn.Conv2d(384, 256, kernel_size=(1, 1), padding=(0, 0)),#(N,256,1,32)

            # nn.BatchNorm2d(256),

            # nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),


        )

        self.fc = nn.Sequential(
            nn.Linear(7680, 256 * 10),
            # nn.Linear(256 * 15, 256 * 5),
            # nn.Linear(256 * 14, 256 * 10),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),#0.3
            nn.BatchNorm1d(256 * 10),

        )
        self.hidden =2
        #2层512个神经元
        #序列长度为2560  每个特征为1
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.hidden, num_layers=2, dropout=0.2)  # 输入维度2560，隐藏层节点数（特征数）30，LSTM有1层
        #lstm_input = (1, 50, 2560)  # batch_size为50

        self.output_layer = nn.Linear(self.hidden, 1)

    def forward(self, x):
        x = self.features(x)
        # print("feature size:", x.size())
        # x = x.view(x.size(0), self.num_flat_features(x))
        # print(x.size())
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc(x)
        lstm_input = x.reshape((-1, 2560, 1))
        output, (hn, cn) = self.lstm(lstm_input)
        output = output.reshape((2560,-1,self.hidden))
        output = self.output_layer(output).squeeze()
        #print(x.size())
        return output.transpose(0,1)

