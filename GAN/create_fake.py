#from main3 import Generator, Discriminator
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image
from torch import nn
import pandas as pd
import os
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(100, 256),  # 用线性变换将输入映射到256维
            nn.ReLU(True),  # relu激活
            nn.Linear(256, 256),  # 线性变换
            nn.ReLU(True),  # relu激活
            nn.Linear(256, 251),  # 线性变换
            nn.Tanh()  # Tanh激活使得生成数据分布在【-1,1】之间，因为输入的真实数据的经过transforms之后也是这个分布
        )

    def forward(self, x):
        x = self.gen(x)
        return x

def test_G(my_class,num):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Tensor = torch.cuda.FloatTensor
    g= generator()
    g.load_state_dict(torch.load('./model/generator_{}.pth'.format(my_class)))
    # d = torch.load('d.pth')
    g = g.to(device)
    # d = d.to(device)
    z = Variable(torch.randn(num, 100)).cuda()
    gen_imgs = g(z)  # 生产图片
    from matplotlib  import pyplot as plt
    gen_imgs = gen_imgs.cpu().detach()
    gen_imgs = gen_imgs.numpy()
    n = len(gen_imgs[0])
    gen_imgs =gen_imgs.T
    if my_class==0:
        plt.subplot(411)
        label=np.array([[1, 0, 0, 0, 0]])
        plt.plot(gen_imgs)
    elif my_class==1:
        plt.subplot(412)
        plt.plot( gen_imgs)
        label = np.array([[0, 1, 0, 0, 0]])
    elif my_class==2:
        plt.subplot(413)
        plt.plot( gen_imgs)
        label = np.array([[0, 0, 1, 0, 0]])
    elif my_class==3:
        plt.subplot(414)
        plt.plot(gen_imgs)
        label = np.array([[0, 0, 0, 1, 0]])

    label=np.tile(label,[num,1] )
    return gen_imgs.T,label

def normal_class(my_class,num):
    res = pd.read_csv('/tmp/ECG-GAN/TEST/result2/Data.txt', header=None)
    label = pd.read_csv('/tmp/ECG-GAN/TEST/result2/Label.txt', header=None)
    resArr = np.array(res)
    labelArr = np.array(label)
    index = np.where(labelArr == my_class)
    resArr = resArr[index[0], :]
    labelArr = labelArr[index[0], :]
    if my_class==0:
        plt.subplot(411)
        plt.plot(resArr[:num,:].T)
    elif my_class==1:
        plt.subplot(412)
        plt.plot( resArr[:num,:].T)
    elif my_class==2:
        plt.subplot(413)
        plt.plot( resArr[:num,:].T)
    elif my_class==3:
        plt.subplot(414)
        plt.plot(resArr[:num,:].T)


def normal(num):
    f, ax = plt.subplots(4, 1)
    f.suptitle('Real_signals')
    ax[0].set_title("N")
    ax[1].set_title("S")
    ax[2].set_title("V")
    ax[3].set_title("F")

    for i in range(0, 4):
        normal_class(i,num)
    plt.show()
    plt.savefig("./model/Real_img.jpg")

#num为1*4的list
def fake(num):
    f, ax = plt.subplots(4, 1)
    f.suptitle('Fake_signals')
    ax[0].set_title("N")
    ax[1].set_title("S")
    ax[2].set_title("V")
    ax[3].set_title("F")

    gen_imgs_N,Label  = test_G(0,num[0])
    gen_imgs_S,label_S  = test_G(1, num[1])
    gen_imgs_V,label_V  = test_G(2, num[2])
    gen_imgs_F,label_F  = test_G(3, num[3])

    gen_imgs_N = np.array(gen_imgs_N)
    gen_imgs_S = np.array(gen_imgs_S)
    gen_imgs_V = np.array(gen_imgs_V)
    gen_imgs_F = np.array(gen_imgs_F)
    Data = gen_imgs_N

    if gen_imgs_S.size != 0:
        Data = np.concatenate((Data, gen_imgs_S), axis=0)
        Label = np.concatenate((Label, label_S), axis=0)
    if gen_imgs_V.size != 0:
        Data = np.concatenate((Data, gen_imgs_V), axis=0)
        Label = np.concatenate((Label, label_V), axis=0)
    if gen_imgs_F.size != 0:
        Data = np.concatenate((Data, gen_imgs_F), axis=0)
        Label = np.concatenate((Label, label_F), axis=0)

    if not os.path.exists('../data_new/fake'):
        os.makedirs('../data_new/fake')

    np.savez('../data_new/fake/fake', ECG_Data=Data, Label=Label)

    plt.show()
    plt.savefig("./model/Fake_img.jpg")



#fake([20,20,20,20])