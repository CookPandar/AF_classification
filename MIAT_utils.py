import numpy as np
import scipy
from scipy.io import loadmat
import pickle
import torch
import os
import random
import math
from torch.utils import data
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.utils.weight_norm import WeightNorm
from sklearn.cluster import KMeans
from itertools import combinations
import matplotlib.pyplot as plt
def get_dataset_for_gan(source_dataset, target_dataset,n_class=5):
    prename = '../data_new'
    if source_dataset == 'DS1':
        DS1_train_data_dir = prename + '/mitdb_uda_DS1/'

    if source_dataset == 'mitdb':
        DS1_train_data_dir = prename + '/mitdb_DS1_DS2_whole_data/'

    if source_dataset == 'svdb':
        DS1_train_data_dir = prename + '/svdb_whole_data/'


    if source_dataset == 'DS1_and_svdb':
        DS1_train_data_dir = prename + '/DS1_and_svdb_data/'


    print('source_train_data_dir:', DS1_train_data_dir)
    DS1_train_data_list = os.listdir(DS1_train_data_dir)


    batch_size = 128
    nb_epoch = 200 - 100
    beat_length = 251
    feature_num = 1
    channel = 1
    class_num = 5
    lr = 0.001

    source_x = np.array([]).reshape((0, feature_num, beat_length, 1))
    source_y = np.array([]).reshape((0, class_num))

    for rec in DS1_train_data_list:
        a = np.load(DS1_train_data_dir + rec)
        beat = a['ECG_Data']
        Label = a['Label']
        source_x = np.concatenate((source_x, beat), axis=0)
        source_y = np.concatenate((source_y, Label), axis=0)


    ##对DS1的数据进行shuffle
    index = np.arange(source_x.shape[0])
    np.random.shuffle(index)
    source_x = source_x[index]
    source_y = source_y[index]

    num2char = {0: 'N', 1: 'S', 2: 'V', 3: 'F', 4: 'Q'}

    num_class = []
    source_x = source_x.astype(np.float32)
    source_y = source_y.astype(np.int64)

    source_x = np.transpose(source_x, (0, 3, 1, 2))


    return source_x, source_y

#fake标识是否需要数据增强
def get_dataset(source_dataset, target_dataset, n_class=5,fake=False):
    train_record_list = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203,
                         205, 207, 208, 209, 215, 220, 223, 230]
    test_record_list = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214,
                        219, 221, 222, 228, 231, 232, 233, 234]
    prename = 'data_new'
    fake_data_list= prename +'/fake'
    #fake_data_list = os.listdir(fake_data_list)
    if source_dataset == 'DS1':
        DS1_train_data_dir = prename+'/mitdb_uda_DS1/'

        if target_dataset == 'DS2':
            DS2_train_data_dir = prename+'/mitdb_uda_DS2/DS2_train/'
            DS2_test_data_dir = prename+'/mitdb_uda_DS2/DS2_test/'

        if target_dataset == 'svdb':
            DS2_train_data_dir = prename+'/svdb_uda_128hz/train/'
            DS2_test_data_dir = prename+'/svdb_uda_128hz/test/'


    if source_dataset == 'mitdb':
        DS1_train_data_dir = prename+'/mitdb_DS1_DS2_whole_data/'
        if target_dataset == 'svdb':
            DS2_train_data_dir = prename+'/svdb_uda_128hz/train/'
            DS2_test_data_dir = prename+'/svdb_uda_128hz/test/'


    if source_dataset == 'svdb':
        DS1_train_data_dir = prename+'/svdb_whole_data/'
        if target_dataset == 'mitdb':
            DS2_train_data_dir = prename+'/mitdb_uda_360hz/train/'
            DS2_test_data_dir = prename+'/mitdb_uda_360hz/test/'

    if source_dataset == 'DS1_and_svdb':
        DS1_train_data_dir = prename+'/DS1_and_svdb_data/'
        if target_dataset == 'DS2':
            DS2_train_data_dir = prename+'/mitdb_uda_DS2/DS2_train/'
            DS2_test_data_dir = prename+'/mitdb_uda_DS2/DS2_test/'
        if target_dataset == 'svdb':
            DS2_train_data_dir = prename+'/svdb_uda_128hz/train/'
            DS2_test_data_dir = prename+'/svdb_uda_128hz/test/'

    print('source_train_data_dir:', DS1_train_data_dir)
    print('target_train_data_dir:', DS2_train_data_dir)
    print('target_test_data_dir:', DS2_test_data_dir)
    DS1_train_data_list = os.listdir(DS1_train_data_dir)
    DS2_train_data_list = os.listdir(DS2_train_data_dir)
    DS2_test_data_list = os.listdir(DS2_test_data_dir)

    batch_size = 128
    nb_epoch = 200 - 100
    beat_length = 251
    feature_num = 1
    channel = 1
    class_num = 5
    lr = 0.001

    dev_x = np.array([]).reshape((0, feature_num, beat_length, 1))
    dev_y = np.array([]).reshape((0, class_num))

    source_x = np.array([]).reshape((0, feature_num, beat_length, 1))
    source_y = np.array([]).reshape((0, class_num))

    target_x = np.array([]).reshape((0, feature_num, beat_length, 1))
    target_y = np.array([]).reshape((0, class_num))

    test_x = np.array([]).reshape((0, feature_num, beat_length, 1))
    test_y = np.array([]).reshape((0, class_num))

    for rec in DS1_train_data_list:
        a = np.load(DS1_train_data_dir + rec)
        beat = a['ECG_Data']
        Label = a['Label']
        source_x = np.concatenate((source_x, beat), axis=0)
        source_y = np.concatenate((source_y, Label), axis=0)

    #读取fake数据
    if fake:
        for rec in fake_data_list:
            a = np.load(fake_data_list + '/fake.npz')
            beat = np.expand_dims(a['ECG_Data'],1)
            beat = np.expand_dims(beat, 3)
            Label = a['Label']
            source_x = np.concatenate((source_x, beat), axis=0)
            source_y = np.concatenate((source_y, Label), axis=0)



    for rec in DS2_train_data_list:
        a = np.load(DS2_train_data_dir + rec)
        beat = a['ECG_Data']
        Label = a['Label']

        target_x = np.concatenate((target_x, beat), axis=0)
        target_y = np.concatenate((target_y, Label), axis=0)
        # print('DS2 TRAIN:', rec)

    for rec in DS2_test_data_list:
        a = np.load(DS2_test_data_dir + rec)
        beat = a['ECG_Data']
        Label = a['Label']

        test_x = np.concatenate((test_x, beat), axis=0)
        test_y = np.concatenate((test_y, Label), axis=0)


    ##对DS1的数据进行shuffle
    index = np.arange(source_x.shape[0])
    np.random.shuffle(index)
    source_x = source_x[index]
    source_y = source_y[index]


    #从DS1中划分验证集
    x_train = source_x
    y_train = source_y
    source_x = x_train[:int(x_train.shape[0] * 0.8)]
    source_y = y_train[:int(x_train.shape[0] * 0.8)]
    dev_x = x_train[int(x_train.shape[0] * 0.8):]
    dev_y = y_train[int(x_train.shape[0] * 0.8):]

    num2char = {0: 'N', 1: 'S', 2: 'V', 3: 'F', 4: 'Q'}

    num_class = []

    for i in range(n_class):
        print('DS1训练集里' + num2char[i] + '类的数量：', int(sum(source_y[:, i])))
        num_class.append(sum(source_y[:, i]))
    print('\n')
    for i in range(n_class):
        print('DS1验证集里' + num2char[i] + '类的数量：', int(sum(dev_y[:, i])))
    print('\n')

    for i in range(n_class):
        print('DS2训练集里' + num2char[i] + '类的数量：', int(sum(target_y[:, i])))
    print('\n')

    for i in range(n_class):
        print('DS2测试集里' + num2char[i] + '类的数量：', int(sum(test_y[:, i])))
    print('\n')

    print('source_x.shape:', source_x.shape)
    print('source_y.shape:', source_y.shape)
    print('dev_x.shape:', dev_x.shape)
    print('dev_y.shape:', dev_y.shape)
    print('target_x.shape:', target_x.shape)
    print('target_y.shape:', target_y.shape)
    print('test_x.shape:', test_x.shape)
    print('test_y.shape:', test_y.shape)
    print('\n')

    source_x = source_x.astype(np.float32)
    source_y = source_y.astype(np.int64)
    target_x = target_x.astype(np.float32)
    target_y = target_y.astype(np.int64)

    dev_x = dev_x.astype(np.float32)
    dev_y = dev_y.astype(np.int64)
    test_x = test_x.astype(np.float32)
    test_y = test_y.astype(np.int64)

    source_x = np.transpose(source_x, (0, 3, 1, 2))
    dev_x = np.transpose(dev_x, (0, 3, 1, 2))
    test_x = np.transpose(test_x, (0, 3, 1, 2))
    target_x = np.transpose(target_x, (0, 3, 1, 2))

    x_train, y_train, x_val, y_val, x_test, y_test, x_target, y_target, num_class = map(torch.tensor,
                                                                                        (source_x, source_y, dev_x,
                                                                                         dev_y, test_x, test_y,
                                                                                         target_x, target_y, num_class))

    y_train = torch.argmax(y_train, dim=1)
    y_val = torch.argmax(y_val, dim=1)
    y_test = torch.argmax(y_test, dim=1)
    y_target = torch.argmax(y_target, dim=1)

    x_train_N = x_train[y_train == 0]
    x_train_S = x_train[y_train == 1]
    x_train_V = x_train[y_train == 2]
    x_train_F = x_train[y_train == 3]
    x_train_Q = x_train[y_train == 4]

    class_center = [x_train_N.mean(dim=0).numpy(), x_train_S.mean(dim=0).numpy(), x_train_V.mean(dim=0).numpy(),
                    x_train_F.mean(dim=0).numpy(), x_train_Q.mean(dim=0).numpy()]


    return x_train, y_train, x_val, y_val, x_test, y_test, x_target, y_target, class_center



def visualize(feat, labels, epoch):
    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.clf()
    for i in range(10):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    plt.xlim(xmin=-8, xmax=8)
    plt.ylim(ymin=-8, ymax=8)
    plt.text(-7.8, 7.3, "epoch=%d" % epoch)
    plt.savefig('./images/epoch=%d.jpg' % epoch)
    plt.draw()
    plt.pause(0.001)



def judge_entropy_criterion(data, pred1, pred2, feature, thresh=None, num_class=5):

    def softmax_entropy(x):
        totalSum = np.sum(np.exp(x), axis=-1,keepdims=True)
        pred = np.exp(x) / totalSum
        ent = -pred * np.log(pred + 1e-6)
        return ent.sum(1)


    print('thresh:', thresh)
    thresh1 = []
    data = data.numpy()
    pred1 = pred1.numpy()
    pred2 = pred2.numpy()

    num = pred1.shape[0]

    new_data = []
    new_label = []
    label_proba = []
    N_data = []
    S_data = []
    V_data = []
    F_data = []
    Q_data = []

    for i in range(num):
        cand_data = data[i, :, :, :]
        label_data = np.zeros((1, num_class))
        ind1 = np.argmax(pred1[i, :])
        value1 = np.max(pred1[i, :])
        ind2 = np.argmax(pred2[i, :])
        value2 = np.max(pred2[i, :])

        if ind1 == ind2:
            new_label.append(ind1)
            new_data.append(cand_data)
            '''存储每个样本的熵指数'''
            # label_proba.append((pred1[i, :]+pred2[i, :])/2)
            '''存储每个样本的置信度'''
            label_proba.append((value1 + value2) / 2)

    new_label = np.array(new_label)
    new_data = np.array(new_data)
    # label_proba = np.array(softmax_entropy(label_proba))
    label_proba = np.array(label_proba)

    # print(new_data.shape, new_label.shape, label_proba.shape)

    pseudo_data = np.array([]).reshape((0, 1, 3, 128)).astype(np.float32)
    pseudo_label = np.array([]).reshape((0)).astype(np.int64)

    for i in range(num_class):
        class_list = np.array(np.where(new_label == i))[0]
        if len(class_list)==0:
            continue
        # print('class_list shape:',class_list.shape)
        temp_ent = label_proba[class_list]
        temp_data = new_data[class_list]
        temp_label = new_label[class_list]
        # ratio = int(thresh * len(temp_label)+1)
        ratio = int(np.ceil(thresh * len(temp_label)))
        ##将置信度降序排列
        ent_sort = np.argsort(temp_ent)[::-1]
        ##取置信度最大的前ratio个样本
        ent_sort = ent_sort[:ratio]
        thresh1.append(temp_ent[-1])
        # print('ent_sort:',ent_sort)
        temp_data = temp_data[ent_sort]
        temp_label = temp_label[ent_sort]
        # print(temp_data.shape, temp_label.shape)
        pseudo_data = np.concatenate((pseudo_data,temp_data),axis=0)
        pseudo_label = np.concatenate((pseudo_label,temp_label),axis=0)

    print('thresh1:', thresh1)
    # print('pseudo data:',pseudo_data.shape)
    # print('pseudo label:', pseudo_label.shape)
    New_Data, New_Label = torch.tensor(pseudo_data), torch.tensor(pseudo_label)
    return New_Data, New_Label, New_Label



def select_class(prob, labels, data, num_class=5, per_class=None):
    labeled = []

    unlabels = []
    norm_factor = []
    for i in range(num_class):
        # class_list = np.array(np.where(classes == i))
        class_list = np.array(np.where(labels == i))
        class_list = class_list[0]

        if len(class_list) != 0:
            class_data = prob[class_list]
            class_prob = np.argsort(-(class_data[:, i]))
            # print(' class_prob:', class_data[:, i])
            # print('class_prob_sort:', class_prob)
            ##选择置信度从高到低前k%的样本
            norm_factor.append(-np.log(class_data[class_prob[int((len(class_prob) - 1) * per_class)], i]))
            # class_prob_topk = class_prob[:int(len(class_prob) * per_class[i])]
            # select_class_index = class_list[class_prob_topk]

            # class_ind = labels[np.where(classes == i), :]
            # rands = np.random.permutation(len(class_list))
            # unlabels.append(class_list[rands[per_class:]])
            # labeled.append(class_list[rands[:per_class]])
            # label_i = np.zeros((per_class, num_class))
            # label_i[:, i] = 1
            # new_label = np.concatenate((new_label, labels[select_class_index]), axis=0)
            # new_data = np.concatenate((new_data, data[select_class_index]), axis=0)
    copy_num = num_class - len(norm_factor)
    for i in range(copy_num):
        norm_factor.append(100)
    # print('norm factor:',norm_factor)
    # prob1 = np.zeros((prob.shape[0], prob.shape[1]))
    # for i in range(num_class):
    #     prob1[:, i] = prob[:, i] / np.exp(-norm_factor[i])

    new_label = []
    new_data = []
    for i in range(data.shape[0]):
        cand_data = data[i, :, :, :]
        ind1 = np.argmax(prob[i, :])
        if prob[i, ind1] > np.exp(-norm_factor[ind1]):
            new_label.append(ind1)
            new_data.append(cand_data)

    new_data, new_label = np.array(new_data), np.array(new_label)

    # unlabel_ind = []

    # for t in unlabels:
    #     for i in t:
    #         unlabel_ind.append(i)
    # label_ind = []
    # for t in new_label:
    #     for i in t:
    #         label_ind.append(i)
    # new_data = np.array(new_data)
    # new_label = np.array(label_ind)
    # print(new_data.shape)
    # print(new_label.shape)
    # unlabel_data = data[unlabel_ind, :, :, :]
    # labeled_data = data[label_ind, :, :, :]
    # train_label = np.array(train_label).reshape((num_class * per_class, num_class))
    # return np.array(labeled_data), np.array(train_label), unlabel_data
    return new_data, new_label, norm_factor



def judge_EHTS(data, pred1, pred2, t_feature, s_feature, s_label, thresh=None, num_class=5):

    def cos_distance(a, b):
        '''a.shape:(batch,fea_dim)
           b.shape:(1, fea_dim)'''

        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1)
        similiarity = np.dot(a, b.T) / (a_norm * b_norm)

        return similiarity

    print('thresh:', thresh)
    data = data.numpy()
    t_feature = t_feature.numpy()
    s_feature = s_feature.numpy()
    s_label = s_label.numpy()
    num = data.shape[0]


    new_label_index = []

    for i in range(num):
        cand_data = data[i, :, :, :]
        ind1 = np.argmax(pred1[i, :])
        ind2 = np.argmax(pred2[i, :])
        if ind1 == ind2:
            new_label_index.append(i)

    New_Label_indx = np.array(new_label_index).astype(np.int64)

    t_feature = t_feature[New_Label_indx]
    data = data[New_Label_indx]

    s_center = []
    probs = []
    for i in range(0, num_class):
        cls_idx = np.argwhere(s_label == i)
        cls_center = s_feature[cls_idx].mean(axis=0)
        probs.append(cos_distance(t_feature, cls_center).reshape(-1))
    probs = np.array(probs).T
    print('tgt probs shape:', probs.shape)
    new_data = []
    new_label = []
    for i in range(t_feature.shape[0]):
        cand_data = data[i, :, :, :]
        ind = np.argmax(probs[i, :])
        value = np.max(probs[i, :])
        if value > thresh:
            new_data.append(cand_data)
            new_label.append(ind)

    New_Data = np.array(new_data).astype(np.float32)
    New_Label = np.array(new_label).astype(np.int64)
    New_Data, New_Label = torch.tensor(New_Data), torch.tensor(New_Label)

    return New_Data, New_Label


def judge_self_paced_learning(data, pred1, pred2, feature, label_ratio, num_class=5):

    data = data.numpy()
    pred1 = pred1.numpy()
    pred2 = pred2.numpy()
    feature = feature.numpy()
    num = pred1.shape[0]
    new_ind = []
    new_data = []
    new_label = []
    new_label_index = []
    select_prob = []
    select_conf = []
    N_data = []
    S_data = []
    V_data = []
    F_data = []
    Q_data = []

    cls_thresh = 0.99*np.ones(num_class)

    if label_ratio >= 1:
        label_ratio = 0.99
        # cls_thresh = 0.01*np.ones(num_class)

    for i in range(num):
        cand_data = data[i, :, :, :]
        label_data = np.zeros((1, num_class))
        ind1 = np.argmax(pred1[i, :])
        value1 = np.max(pred1[i, :])
        ind2 = np.argmax(pred2[i, :])
        value2 = np.max(pred2[i, :])
        avg_prob = (pred1[i, :] + pred1[i, :]) / 2
        avg_conf = (value1 + value2) / 2
        #avg_prob为二者均一致情况下，未softmax的 num_classes 维向量
        if ind1 == ind2 and ind1 == 0:
            new_label.append(ind1)
            select_conf.append(avg_conf)
            select_prob.append(avg_prob)
            new_data.append(cand_data)

        if ind1 == ind2 and ind1 == 1:
            new_label.append(ind1)
            select_conf.append(avg_conf)
            select_prob.append(avg_prob)
            new_data.append(cand_data)

        if ind1 == ind2 and ind1 == 2:
            new_label.append(ind1)
            select_conf.append(avg_conf)
            select_prob.append(avg_prob)
            new_data.append(cand_data)

        if ind1 == ind2 and ind1 == 3:
            new_label.append(ind1)
            select_conf.append(avg_conf)
            select_prob.append(avg_prob)
            new_data.append(cand_data)

        if ind1 == ind2 and ind1 == 4:
            new_label.append(ind1)
            select_conf.append(avg_conf)
            select_prob.append(avg_prob)
            new_data.append(cand_data)

    New_Data, New_Label, Select_Prob, Select_Conf = np.array(new_data), np.array(new_label), np.array(select_prob), \
                                                    np.array(select_conf)
    Ft_Prob = Select_Prob
    for idx_cls in range(num_class):
        sample_id = np.argwhere(New_Label == idx_cls)[:, 0]

        sample_conf = list(Select_Conf[sample_id])

        if len(sample_conf) == 0:
            continue
        sample_conf.sort(reverse=True)  # sort in descending order
        len_cls = len(sample_conf)
        len_cls_thresh  = int(math.floor(len_cls * label_ratio))
        # print('len_cls_thresh:', len_cls_thresh)
        if len_cls_thresh == 0:
            continue
        cls_thresh[idx_cls] = sample_conf[len_cls_thresh - 1]

    print('label ratio: {:.4f}'.format(label_ratio))
    print('cbst thresh:', [t.round(4) for t in cls_thresh])
    Select_Prob = Select_Prob / cls_thresh
    select_label = []
    select_data = []
    for i in range(Select_Prob.shape[0]):
        if np.max(Select_Prob[i, :]) >= 1:
            label = np.argmax(Select_Prob[i, :])
            select_label.append(label)
            select_data.append(New_Data[i])

    New_Data = np.array(select_data).astype(np.float32)
    New_Label = np.array(select_label).astype(np.int64)

    New_Data, New_Label = torch.tensor(New_Data), torch.tensor(New_Label)

    del new_ind, new_data, new_label, new_label_index, select_prob, select_conf

    return New_Data, New_Label,Ft_Prob


def judge_func_my(data, pred1, pred2, feature, thresh=None, num_class=5):
    data = data.numpy()
    pred1 = pred1.numpy()
    pred2 = pred2.numpy()
    feature = feature.numpy()

    num = pred1.shape[0]
    if not thresh:
        thresh = [0.85, 0.85, 0.75, 0.65, 0.5]
    print('thresh:', thresh)

    new_ind = []
    new_data = []
    new_label = []
    new_label_index = []
    label_proba = []
    N_data = []
    S_data = []
    V_data = []
    F_data = []
    Q_data = []

    for i in range(num):
        cand_data = data[i, :, :, :]
        ind1 = np.argmax(pred1[i, :])
        value1 = np.max(pred1[i, :])
        ind2 = np.argmax(pred2[i, :])
        value2 = np.max(pred2[i, :])

        # if ind1 == ind2 and ind1 == 0:
        if ind2 == 0:
            # if max(value1, value2) > thresh[0]:  # 0.85
            if value2 > thresh[0]:
                new_ind.append(ind2)
                new_data.append(cand_data)
                new_label_index.append(i)

        if ind2 == 1:
            if value2 > thresh[1]:  # 0.85
                new_ind.append(ind2)
                new_data.append(cand_data)
                new_label_index.append(i)

        if ind2 == 2:
            if value2 > thresh[2]:  # 0.75
                new_ind.append(ind2)
                new_data.append(cand_data)
                new_label_index.append(i)

        if ind2 == 3:
            if value2 > thresh[3]:  # 0.65
                new_ind.append(ind2)
                new_data.append(cand_data)
                new_label_index.append(i)
        if ind2 == 4:
            if value2 > thresh[4]:
                new_ind.append(ind2)
                new_data.append(cand_data)
                new_label_index.append(i)



    New_Data, New_Label, New_Label_index = np.array(new_data), np.array(new_ind), np.array(new_label_index)
    New_Data = New_Data.astype(np.float32)
    New_Label = New_Label.astype(np.int64)
    New_Label_index = New_Label_index.astype(np.int64)

    New_Data, New_Label = torch.tensor(New_Data), torch.tensor(New_Label)
    New_Label_index = torch.tensor(New_Label_index)

    return New_Data, New_Label


def judge_func(data, pred1, pred2, feature, thresh=None, num_class=5):
    # print('thresh:', [round(t, 4) for t in thresh])

    P_H = 0.07
    P_L = 0.007
    upper = 0.9999
    lower = 0.01
    step = 0.01
    # if not thresh:
    #     thresh = [0.85, 0.85, 0.75, 0.65, 0.5]
    if thresh is None:
        thresh = [0.85, 0.85, 0.75, 0.65, 0.5]
        # thresh = [0.85, 0.85, 0.85, 0.85, 0.85]
    else:
        thresh = [thresh] * 5

    # print('thresh:', thresh)
    # print('thresh:', [t.round(4) for t in thresh])

    data = data.numpy()
    pred1 = pred1.numpy()
    pred2 = pred2.numpy()
    feature = feature.numpy()

    num = pred1.shape[0]

    new_ind = []
    new_data = []
    new_label = []
    new_label_index = []
    label_proba = []
    N_data = []
    S_data = []
    V_data = []
    F_data = []
    Q_data = []
    new_prob =[]


    for i in range(num):
        cand_data = data[i, :, :, :]
        label_data = np.zeros((1, num_class))
        ind1 = np.argmax(pred1[i, :])
        value1 = np.max(pred1[i, :])
        ind2 = np.argmax(pred2[i, :])
        value2 = np.max(pred2[i, :])

        if ind1 == ind2 and ind1 == 0:
            if max(value1, value2) > thresh[0]:  # 0.85
                label_data[0, ind1] = 1
                new_ind.append(ind1)
                label_proba.append(max(value1, value2))
                new_data.append(cand_data)
                new_label_index.append(i)
                ave_fx = (pred1[i, :]+pred2[i, :])/2
                new_prob.append(ave_fx)

        if ind1 == ind2 and ind1 == 1:
            if max(value1, value2) > thresh[1]:  # 0.85
                label_data[0, ind1] = 1
                new_ind.append(ind1)
                label_proba.append(max(value1, value2))
                new_data.append(cand_data)
                new_label_index.append(i)
                ave_fx = (pred1[i, :]+pred2[i, :])/2
                new_prob.append(ave_fx)

        if ind1 == ind2 and ind1 == 2:
            if max(value1, value2) > thresh[2]:  # 0.75
                label_data[0, ind1] = 1
                new_ind.append(ind1)
                label_proba.append(max(value1, value2))
                new_data.append(cand_data)
                new_label_index.append(i)
                ave_fx = (pred1[i, :]+pred2[i, :])/2
                new_prob.append(ave_fx)

        if ind1 == ind2 and ind1 == 3:
            if max(value1, value2) > thresh[3]:  # 0.65
                label_data[0, ind1] = 1
                new_ind.append(ind1)
                label_proba.append(max(value1, value2))
                new_data.append(cand_data)
                new_label_index.append(i)
                ave_fx = (pred1[i, :]+pred2[i, :])/2
                new_prob.append(ave_fx)

        if ind1 == ind2 and ind1 == 4:
            if max(value1, value2) > thresh[4]:  # 0.2
                label_data[0, ind1] = 1
                new_ind.append(ind1)
                label_proba.append(max(value1, value2))
                new_data.append(cand_data)
                new_label_index.append(i)
                ave_fx = (pred1[i, :]+pred2[i, :])/2
                new_prob.append(ave_fx)


    New_Data, New_Label, New_Label_Prob = np.array(new_data), np.array(new_ind), np.array(label_proba)
    New_Label_Index = np.array(new_label_index).astype(np.int64)
    New_Data = New_Data.astype(np.float32)
    New_Label = New_Label.astype(np.int64)
    New_Label_Prob = New_Label_Prob.astype(np.float32)

    New_Data, New_Label = torch.tensor(New_Data), torch.tensor(New_Label)
    New_Label_Prob = torch.tensor(New_Label_Prob)
    New_Label_Index = torch.tensor( New_Label_Index)

    return New_Data, New_Label, new_prob


def balance_batch_generator(data, batch_size, shuffle=True, test=False, f1_cls=None):
    # if shuffle:
    #     data = shuffle_aligned_list(data)
    # batch_count = 0
    #######################################
    # Generate balanced labeled source examples.
    # Only used on large dataset as
    # the training set is quite unbalanced.
    #######################################
    ecg_data = data[0]
    label = data[1]

    while True:
        data_batch_x = []
        data_batch_y = []
        # random.seed(666)
        for i in range(4):
            idx = torch.nonzero(torch.eq(label, i)).numpy().reshape(-1)
            # print('idx:',idx)
            # print('len idx:', len(idx))
            inds = random.sample(list(idx), min(batch_size // 4, len(idx)))
            data_batch_x.append(ecg_data[inds])
            data_batch_y.append(label[inds])
        data_batch_x = torch.cat(data_batch_x, dim=0)
        data_batch_y = torch.cat(data_batch_y, dim=0)
        # print(data_batch_x.shape, data_batch_y.shape,)

        yield [data_batch_x, data_batch_y]



def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]

#产生多组批大小的数据
def batch_generator(data, batch_size, shuffle=True, test=False, triplet=False):

    if shuffle:
        data = shuffle_aligned_list(data)
    batch_count = 0
    while True:
        if test:
            if batch_count * batch_size >= len(data[0]):
                batch_count = 0
                if shuffle:
                    data = shuffle_aligned_list(data)
        else:
            if batch_count * batch_size + batch_size >= len(data[0]):
                batch_count = 0
                if shuffle:
                    data = shuffle_aligned_list(data)
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]


def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    labels_one_hot = np.zeros((len(labels_dense), num_classes))
    labels_dense = list(labels_dense)
    for i, t in enumerate(labels_dense):
        labels_one_hot[i, t] = 1
    return labels_one_hot


class FocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class=5, alpha=0.25, gamma=2, balance_index=1, smooth=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        epsilon = 1e-6
        logit = F.softmax(input, dim=1)
        logit = torch.clamp(logit, epsilon, 1 - epsilon)

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))

        target = target.view(-1, 1)


        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(one_hot_key, self.smooth, 1.0 - self.smooth)
        # pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = one_hot_key * torch.log(logit)
        # logpt = pt.log()

        # alpha = alpha[idx]
        # loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt
        loss = - torch.pow((1 - logit), self.gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


def focal_loss_zhihu(inputs, target, gamma=1):#0.5
    '''
    :param input: 使用知乎上面大神给出的方案  https://zhuanlan.zhihu.com/p/28527749
    :param target:
    :return:
    '''

    def compute_class_weights(histogram):
        classWeights = np.ones(5, dtype=np.float32)
        normHist = histogram / np.sum(histogram)
        for i in range(5):
            classWeights[i] = 1 / (np.log(1.10 + normHist[i]))
        return classWeights

    # target = target.long()
    #
    # number_0 = torch.sum(target == 0).item()
    # number_1 = torch.sum(target == 1).item()
    # number_2 = torch.sum(target == 2).item()
    # number_3 = torch.sum(target == 3).item()
    # number_4 = torch.sum(target == 4).item()
    #
    #
    # frequency = torch.tensor((number_0, number_1, number_2, number_3, number_4), dtype=torch.float32)
    # frequency = frequency.numpy()
    # classWeights = compute_class_weights(frequency)
    #
    # weights = torch.from_numpy(classWeights).float()
    # weights = weights[target.view(-1)]  # 这行代码非常重要
    # weights = weights.to(inputs.device)
    epsilon = 1e-10
    P = F.softmax(inputs, dim=-1)
    P = torch.clamp(P, epsilon, 1-epsilon)
    # print('P:',P)# shape [num_samples,num_classes]
    target = target.view(-1, 1)
    idx = target.cpu().long()
    one_hot_key = torch.FloatTensor(target.size(0), 5).zero_()
    one_hot_key = one_hot_key.scatter_(1, idx, 1)
    if one_hot_key.device != inputs.device:
        one_hot_key = one_hot_key.to(inputs.device)

    # class_mask = inputs.data.new(N, C).fill_(0)
    # class_mask = Variable(class_mask)
    # ids = target.view(-1, 1)
    # class_mask.scatter_(1, ids.data, 1.)#shape [num_samples,num_classes]  one-hot encoding

    probs = (P * one_hot_key).sum(1).view(-1, 1)  # shape [num_samples,]
    log_p = probs.log()


    # print('in calculating batch_loss', weights.shape, probs.shape, log_p.shape)

    # batch_loss = -weights * (torch.pow((1 - probs), gamma)) * log_p
    batch_loss = -(torch.pow((1 - probs), gamma)) * log_p


    # print(batch_loss.shape)

    loss = batch_loss.mean()
    return loss


class LDAMLoss(nn.Module):

    '''paper:Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss'''

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)



def separability_loss(s_feature, t_feature, s_label, t_label, num_classes=5):

    uni = np.array(np.unique(t_label.cpu().numpy(), return_counts=True))
    ##uni=[[0, 2],
    ##     [116, 12]]
    mi = np.min(uni[1])
    if len(uni[1]) < num_classes:
        mi = 0
    ma = np.max(uni[1])
    imbalance_parameter = (mi + 1) / (ma + 1)
    # print('imbalance_parameter:', imbalance_parameter)

    latents = torch.cat((s_feature, t_feature), 0)
    labels = torch.cat((s_label, t_label), 0)

    criteria = nn.CosineEmbeddingLoss()

    loss_up = 0
    one_cuda = torch.ones(1).cuda()
    mean = torch.mean(latents, dim=0).to(s_label.device).view(1, -1)
    loss_down = 0
    for i in range(num_classes):
        indexes = labels.eq(i)
        mean_i = torch.mean(latents[indexes], dim=0).view(1, -1)
        if str(mean_i.norm().item()) != 'nan':
            loss_up += criteria(latents[indexes], mean_i, one_cuda)
            # for latent in latents[indexes]:
            #     loss_up += criteria(latent.view(1, -1), mean_i, one_cuda)
            loss_down += criteria(mean, mean_i, one_cuda)
    loss = (loss_up / loss_down) * imbalance_parameter
    return loss

def extract_embeddings(model, dataloader):
    model.eval()
    n_samples = dataloader.batch_size * len(dataloader)
    embeddings = np.zeros((n_samples, model.n_outputs))
    labels = np.zeros(n_samples)
    k = 0

    for images, target in dataloader:
        with torch.no_grad():
            images = images.cuda()
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)

    return embeddings, labels

def triplet_loss(feature, label, margin=1):
    # model.train()
    # emb = model(batch["X"].cuda())
    # y = batch["y"].cuda()
    emb =  feature
    y = label

    with torch.no_grad():
        triplets = get_triplets(emb, y, margin)
        # print('triplets.shape:',triplets.shape)

    f_A = emb[triplets[:, 0]]
    f_P = emb[triplets[:, 1]]
    f_N = emb[triplets[:, 2]]

    ap_D = (f_A - f_P).pow(2).sum(1)  # .pow(.5)
    an_D = (f_A - f_N).pow(2).sum(1)  # .pow(.5)
    losses = F.relu(ap_D - an_D + margin)

    return losses.mean()

def get_triplets(embeddings, y, margin):
    # margin = 1
    D = pdist(embeddings)
    D = D.cpu()

    y = y.cpu().data.numpy().ravel()
    trip = []
    ap = np.arange(len(embeddings))
    for label in set(y):
        if label != 0:
            label_mask = (y == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                # continue
                # ap = [(label_indices[0],label_indices[0])]
                label_indices=np.array([label_indices[0],label_indices[0]])
            neg_ind = np.where(np.logical_not(label_mask))[0]


            ap = list(combinations(label_indices, 2))  # All anchor-positive pairs
            ap = np.array(ap)

            ap_D = D[ap[:, 0], ap[:, 1]]
            # if len(neg_ind)==0:
            #     continue
            # # GET HARD NEGATIVE
            # if np.random.rand() < 0.5:
            #   trip += get_neg_hard(neg_ind, hardest_negative,
            #                D, ap, ap_D, margin)
            # else:
            trip += get_neg_hard(neg_ind, random_neg, D, ap, ap_D, margin)

    if len(trip) == 0 :
        # ap = ap[0]
        # if len(neg_ind) > 0:
        #     trip.append([ap[0], ap[1], neg_ind[0]])
        # else:
        #     trip.append([ap[0], ap[1], ap[0]])
        trip.append([0, 1, 0])

    trip = np.array(trip)

    return torch.LongTensor(trip)


def get_triplets_minority(embeddings, y):
    margin = 1
    D = pdist(embeddings)
    D = D.cpu()

    y = y.cpu().data.numpy().ravel()
    trip = []
    # ap = np.arange(len(embeddings))
    num_per_cls = {}
    for label in set(y):
        label_mask = (y == label)
        label_indices = np.where(label_mask)[0]
        num_per_cls[label] = len(label_indices)

    minority_cls = min(num_per_cls, key=num_per_cls.get)


    label_mask = (y == minority_cls)
    label_indices = np.where(label_mask)[0]
    # print('minority_cls:', minority_cls)
    # print('num:', len(label_indices))
    if len(label_indices) < 2:
        # continue
        # ap = [(label_indices[0],label_indices[0])]
        label_indices=np.array([label_indices[0],label_indices[0]])
    neg_ind = np.where(np.logical_not(label_mask))[0]

    ap = list(combinations(label_indices, 2))  # All anchor-positive pairs
    ap = np.array(ap)

    ap_D = D[ap[:, 0], ap[:, 1]]
    # if len(neg_ind)==0:
    #     continue
    # # GET HARD NEGATIVE
    # if np.random.rand() < 0.5:
    #   trip += get_neg_hard(neg_ind, hardest_negative,
    #                D, ap, ap_D, margin)
    # else:
    trip += get_neg_hard(neg_ind, random_neg, D, ap, ap_D, margin)

    if len(trip) == 0 :
        ap = ap[0]
        if len(neg_ind) > 0:
            trip.append([ap[0], ap[1], neg_ind[0]])
        else:
            trip.append([ap[0], ap[1], ap[0]])

    trip = np.array(trip)

    return torch.LongTensor(trip)

def pdist(vectors):
    D = -2 * vectors.mm(torch.t(vectors))
    D += vectors.pow(2).sum(dim=1).view(1, -1)
    D += vectors.pow(2).sum(dim=1).view(-1, 1)

    return D


def get_neg_hard(neg_ind,select_func, D, ap, ap_D, margin):
    trip = []
    if len(neg_ind)>0:
        for ap_i, ap_di in zip(ap, ap_D):
            loss_values = (ap_di -D[torch.LongTensor(np.array([ap_i[0]])),torch.LongTensor(neg_ind)] + margin)

            # loss_values = loss_values.data.cpu().numpy()
            loss_values = loss_values.detach().cpu().numpy()
            neg_hard = select_func(loss_values)

            if neg_hard is not None:
                neg_hard = neg_ind[neg_hard]
                trip.append([ap_i[0], ap_i[1], neg_hard])

    return trip

def random_neg(loss_values):
    neg_hards = np.where(loss_values > 0)[0]
    return np.random.choice(neg_hards) if len(neg_hards) > 0 else None

def ContrastiveLoss(s_fea, s_label, t_fea, t_label, margin=10):

    # assert s_fea.shape == t_fea.shape

    '''source to target loss'''
    # intra_mask = torch.zeros(s_fea.shape[0], s_fea.shape[0]).to(s_label.device)
    intra_mask = torch.zeros(s_fea.shape[0], t_fea.shape[0]).to(s_label.device)
    for i in range(len(s_label)):
        idx = torch.nonzero(torch.eq(t_label, s_label[i]))
        intra_mask[i, idx] = 1.
    inter_mask = 1 - intra_mask
    # assert (intra_mask.sum() + inter_mask.sum()) == s_fea.shape[0] ** 2
    assert (intra_mask.sum() + inter_mask.sum()) == s_fea.shape[0] * t_fea.shape[0]

    '''compute dist mat of s_feature and t_feature'''

    vecProd = torch.mm(s_fea, torch.transpose(t_fea, 1, 0))
    SqA = s_fea ** 2
    sumSqA = torch.sum(SqA, dim=1).view(1, -1)
    sumSqAEx = torch.transpose(sumSqA, 1, 0).repeat(1, vecProd.shape[1])

    SqB = t_fea ** 2
    sumSqB = torch.sum(SqB, dim=1).view(1, -1)
    sumSqBEx = sumSqB.repeat(vecProd.shape[0], 1)
    # print('sumSqAEx shape:', sumSqAEx)
    # print('sumSqBEx shape:', sumSqBEx)
    ###plus 1 for dist_mat,防止0开根号出现错误
    ##2020/6/7：之前没有加1
    SqED = sumSqBEx + sumSqAEx - 2 * vecProd
    SqED = torch.clamp(SqED, min=1e-20)
    # print(' vecProd shape:', vecProd)
    # print(' dist_mat shape:', SqED)
    # SqED = (sumSqBEx + sumSqAEx - 2 * vecProd) ** 0.5
    SqED = SqED** 0.5
    dist_mat = SqED.to(s_label.device)

    # m = margin * torch.ones(s_fea.shape[0], s_fea.shape[0]).to(s_label.device)
    m = margin * torch.ones(s_fea.shape[0], t_fea.shape[0]).to(s_label.device)
    intra_loss = ((intra_mask * dist_mat) ** 2).sum() / (intra_mask.sum() + 1)
    # inter_loss = ((inter_mask * torch.max(torch.zeros(s_fea.shape[0], s_fea.shape[0]).to(s_label.device),
    #                                       (m - dist_mat))) ** 2).sum() / (inter_mask.sum() + 1)
    inter_loss = ((inter_mask * torch.max(torch.zeros(s_fea.shape[0], t_fea.shape[0]).to(s_label.device),
                                          (m - dist_mat))) ** 2).sum() / (inter_mask.sum() + 1)
    # print('inter loss :', inter_loss)
    # print('intra loss :', intra_loss)

    # '''target domain loss'''
    # intra_mask_t = torch.zeros(t_fea.shape[0], t_fea.shape[0]).to(t_label.device)
    # for i in range(len(t_label)):
    #     idx = torch.nonzero(torch.eq(t_label, t_label[i]))
    #     intra_mask_t[i, idx] = 1.
    # inter_mask_t = 1 - intra_mask_t
    # assert (intra_mask_t.sum() + inter_mask_t.sum()) == t_fea.shape[0] ** 2
    #
    # vecProd_t = torch.mm(t_fea, torch.transpose(t_fea, 1, 0))
    # SqA_t = t_fea ** 2
    # sumSqA_t = torch.sum(SqA_t, dim=1).view(1, -1)
    # sumSqAEx_t = torch.transpose(sumSqA_t, 1, 0).repeat(1, vecProd_t.shape[1])
    # # print('sumSqAEx shape:', sumSqAEx.shape)
    # SqB_t = t_fea ** 2
    # sumSqB_t = torch.sum(SqB_t, dim=1).view(1, -1)
    # sumSqBEx_t = sumSqB_t.repeat(vecProd_t.shape[0], 1)
    #
    # SqED_t = (sumSqBEx_t + sumSqAEx_t - 2 * vecProd_t+1e-10)**0.5
    # dist_mat_t = SqED_t.to(t_label.device)
    #
    # m_t = margin * torch.ones(t_fea.shape[0], t_fea.shape[0]).to(t_label.device)
    # intra_loss_t = ((intra_mask_t * dist_mat_t)**2).sum() / (intra_mask_t.sum() + 1)
    # inter_loss_t = ((inter_mask_t * torch.max(torch.zeros(t_fea.shape[0], t_fea.shape[0]).to(t_label.device),
    #                                      (m_t - dist_mat_t)))**2).sum() / (inter_mask_t.sum() + 1)
    # print('inter loss t:', inter_loss_t)
    # print('intra loss t:', intra_loss_t)
    return inter_loss + intra_loss




def EuclideanDistances(A, B):
    '''求矩阵A和B两两向量间的欧氏距离'''
    k = 0.1
    P = 3

    BT = B.transpose()
    vecProd = np.dot(A, BT)
    SqA = A ** 2
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))

    SqB = B ** 2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = np.array(sumSqBEx + sumSqAEx - 2 * vecProd)
    # SqED[SqED < 0] = 0.0
    # ED = np.sqrt(SqED)
    a = SqED.flatten()
    b = np.sort(a)
    dist_thresh = b[int(len(a) * k)]
    # print('thresh:', dist_thresh)
    local_density_vector = np.sum(SqED < dist_thresh, axis=1)
    # print(' local_density_vector :', local_density_vector )
    class_center_index = np.argmax(local_density_vector)
    X = SqED[class_center_index].reshape(A.shape[0], 1)

    clf = KMeans(n_clusters=P)
    clf.fit(X)
    labels = clf.labels_
    class_0_density = local_density_vector[labels == 0].sum()
    class_1_density = local_density_vector[labels == 1].sum()
    class_2_density = local_density_vector[labels == 2].sum()
    # print('label0:',labels==0)
    # print('label1:', labels == 1)
    # print('label2:', labels == 2)
    # print([class_0_density, class_1_density, class_2_density])
    class_density_sort = np.argsort([class_0_density, class_1_density, class_2_density])
    # easy_data = A[labels == class_density_sort[2]]
    # mid_data = A[labels == class_density_sort[1]]
    # hard_data = A[labels == class_density_sort[0]]
    # print(A, easy_data, mid_data, hard_data)
    # return  easy_data, mid_data, hard_data
    return labels == class_density_sort[2], labels == class_density_sort[1], labels == class_density_sort[0]



def CrossEntropyLoss(predict_prob, label, normal_factor=None, class_level_weight=None,
                     instance_level_weight=None, epsilon=1e-12):
    N, C = predict_prob.size()
    # N, C = label.size()
    # N_, C_ = predict_prob.size()
    # assert N == N_ and C == C_, 'fatal error: dimension mismatch!'

    # beta = 0.999
    # effective_num = 1.0 - np.power(beta, class_level_weight)
    # weights = (1.0 - beta) / np.array(effective_num)
    # no_of_classes = 5
    # weights = weights / np.sum(weights) * no_of_classes
    # weights = torch.FloatTensor(weights)

    target = label.view(-1, 1)
    idx = target.cpu().long()
    one_hot_key = torch.FloatTensor(target.size(0), 5).zero_()
    labels_one_hot = one_hot_key.scatter_(1, idx, 1).to(label.device)

    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    if normal_factor is None:
        normal_factor = -np.log(np.array([0.2, 0.2, 0.2, 0.2, 0.2]))

    label_loss = torch.mul(torch.tensor(normal_factor).type_as(labels_one_hot).cuda(), labels_one_hot).sum()

    # prob = torch.zeros(N, C).type_as(predict_prob)
    # for i in range(5):
    #     prob[:, i] = predict_prob[:, i] / np.exp(-normal_factor[i])
    #
    # ce = F.cross_entropy(prob, label)

    # prob_1 = F.softmax(prob, dim=1)
    # print('prob:', prob_1)
    # ce = -labels_one_hot * torch.log(prob_1 + epsilon)
    # return torch.sum(instance_level_weight * ce * class_level_weight) / float(N)
    # return torch.mean(torch.sum(ce * weights, 1))
    # return F.cross_entropy(predict_prob, label, weights)
    return label_loss



def EntropyLoss(pred_1, pred_2, class_level_weight=None, instance_level_weight=None, epsilon=1e-20):
    N, C = pred_2.size()
    pred_1 = F.softmax(pred_1, dim=1)
    pred_2 = F.softmax(pred_2, dim=1)
    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    entropy = -pred_1 * torch.log(pred_2 + epsilon)
    return entropy.sum(1).mean()
    # return torch.sum(instance_level_weight * entropy * class_level_weight) / float(N)

def MRKLD_Loss(pred_1, num_class=5, epsilon=1e-6):
    '''Paper: Confidence Regularized Self-Training'''
    pred_1 = F.softmax(pred_1, dim=1)
    entropy = -(1/num_class)* torch.log(pred_1 + epsilon)
    return entropy.sum(1).mean()

class CrossEntropyLabelSmooth(nn.Module):
    '''Cross entropy loss with label smoothing regularizer.
        Reference:
        Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
        Equation: y = (1 - epsilon) * y + epsilon / K.
        Args:
            num_classes (int): number of classes.
            epsilon (float): weight.'''

    def __init__(self, num_classes=5, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        # loss = (- targets * log_probs).mean(0).sum()
        loss = (- targets * log_probs).sum(dim=-1).mean()
        return loss


def symmetric_kl_loss(source_fea, target_fea, epsilon=1e-10):
    '''paper:Adaptive Semi-supervised Learning for Cross-domain Sentiment Classification'''
    a = torch.mean(source_fea, dim=0)
    b = torch.mean(target_fea, dim=0)
    a /= torch.sum(a)
    b /= torch.sum(b)

    a = torch.clamp(a, epsilon, 1)
    b = torch.clamp(b, epsilon, 1)
    # print(a.shape, b.shape)

    loss = torch.sum(a * torch.log(a / b)) + torch.sum(b * torch.log(b / a))
    return loss


def discrepancy_slice_wasserstein(p1, p2):
    '''论文：Sliced Wasserstein Discrepancy for Unsupervised Domain Adaptation'''
    s = p1.shape
    if s[1] > 1:
        proj = torch.randn(s[1], 128).cuda()
        proj *= torch.rsqrt(torch.sum(torch.mul(proj, proj), 0, keepdim=True))
        p1 = torch.matmul(p1, proj)
        p2 = torch.matmul(p2, proj)
    p1 = torch.topk(p1, s[0], dim=0)[0]
    p2 = torch.topk(p2, s[0], dim=0)[0]
    dist = p1 - p2
    wdist = torch.mean(torch.mul(dist, dist))
    return wdist

def Max_BNM_loss(pred, dom_label=None):
    '''论文：Towards Discriminability and Diversity:Batch Nuclear-norm Maximization under Label Insufficient Situations'''
    # print('dom_label:', dom_label )
    # labeled_idx = torch.nonzero(torch.eq(dom_label, 0))
    # print('dom_label 1:', labeled_idx)
    # labeled_loss = F.cross_entropy(pred[labeled_idx], target[labeled_idx])

    # unlabeled_idx = torch.nonzero(torch.eq(dom_label, 1))
    softmax_tgt = nn.Softmax(dim=1)(pred)
    _, s_tgt, _ = torch.svd(softmax_tgt)
    unlabeled_loss = -torch.mean(s_tgt)
    return unlabeled_loss

def mcc_loss(tgt_pred, t=2):
    '''Paper: Minimum Class Confusion for Versatile DomainAdaptation'''
    B = tgt_pred.size(0)
    C = tgt_pred.size(1)
    epsilon = 1e-12
    '''Probability Rescaling'''
    # print('t=1:', F.softmax(tgt_pred))
    tgt_soft = torch.exp(tgt_pred / t)
    tgt_soft = tgt_soft / tgt_soft.sum(1).unsqueeze(1)
    tgt_soft = torch.clamp(tgt_soft , epsilon, 1 - epsilon)
    # print('t=2:', tgt_soft )
    '''Uncertainty Weighting'''
    # ent = - (tgt_soft * torch.log(tgt_soft)).sum(1)
    # print('ent:', ent)
    # exp_ent = B * (1 + torch.exp(-ent))
    # W = torch.zeros(B, B).to(tgt_pred.device)
    # for i in range(B):
    #     W[i, i] = exp_ent[i] /(1 + torch.exp(-ent)).sum()
        # W[i, i] = exp_ent[i] / exp_ent.sum()


    '''Category Normalization.'''

    # C = torch.mm(torch.mm(tgt_soft.transpose(1, 0), W), tgt_soft)
    C = torch.mm(tgt_soft.transpose(1, 0), tgt_soft)
    C = C / (C.sum(1).unsqueeze(1))
    # print('C:', C)
    '''Minimum Class Confusion Loss'''
    loss = C.sum() - (C[0,0] + C[1, 1] + C[2, 2] + C[3, 3])
    return loss


def inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=1000):
    return initial_lr * ((1 + gamma * min(1.0, step / float(max_iter))) ** (- power))

def plot_confusion_matrix():
    import seaborn as sns
    import pandas as pd
    from sklearn import metrics
    conf_mat=pd.DataFrame(np.array(sklearn.metrics.confusion_matrix(y_test.argmax(axis=1), y_pred2.argmax(axis=1))),
                             index = ['0', '1','2', '3','4'],
                            columns=['pred_0', 'pred_1','pred_2', 'pred_3','pred_4'])
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(conf_mat.apply(lambda x: x/x.sum(),axis=1), annot=True, cmap = 'Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def mixup_data_sup(x, y, alpha=1.0):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = np.random.permutation(batch_size)
    # x, y = x.numpy(), y.numpy()
    # mixed_x = torch.Tensor(lam * x + (1 - lam) * x[index,:])
    mixed_x = lam * x + (1 - lam) * x[index, :]
    # y_a, y_b = torch.Tensor(y).type(torch.LongTensor), torch.Tensor(y[index]).type(torch.LongTensor)
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def mixup_data(x, y, alpha=1.0):
    '''Compute the mixup data. Return mixed inputs, mixed target, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = np.random.permutation(batch_size)
    x, y = x.data.cpu().numpy(), y.data.cpu().numpy()

    mixed_x = torch.Tensor(lam * x + (1 - lam) * x[index, :])
    mixed_y = torch.Tensor(lam * y + (1 - lam) * y[index, :])

    # mixed_x = Variable(mixed_x.cuda())
    # mixed_y = Variable(mixed_y.cuda())
    return mixed_x, mixed_y, lam


def  MaxSquareloss(prob):
    """
    paper: Domain Adaptation for Semantic Segmentation with Maximum Squares Loss
    """
    # prob -= 0.5
    prob = F.softmax(prob, dim=1)
    loss = -torch.mean(torch.pow(prob, 2).sum(dim=1),dim=0)
    return loss



class CoralLoss(nn.Module):

    def __init__(self):
        super().__init__()

        # self.dist = distance

    def forward(self, source, target):
        d = source.data.shape[1]

        # source covariance
        xm = torch.mean(source, 0, keepdim=True) - source
        xc = xm.t() @ xm

        # target covariance
        xmt = torch.mean(target, 0, keepdim=True) - target
        xct = xmt.t() @ xmt

        # frobenius norm between source and target
        loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
        loss = loss / (4 * d * d)
        return loss


def sep_loss(s_feature, t_feature, s_label, t_label):
    def get_cos_dis(target, behaviored):

        attention_distribution = 0
        for i in range(behaviored.shape[0]):
            attention_score = torch.cosine_similarity(target, behaviored[i], dim=0)  # 计算每一个元素与给定元素的余弦相似度
            attention_distribution += attention_score
        return attention_distribution

    numerator = 0
    denominator = 0
    from collections import Counter
    label_t = t_label.cpu().detach().numpy()
    label_per_class = []
    for i in np.unique(label_t):
        label_per_class.append(len(np.argwhere(label_t == i)))

    label_per_class = sorted(label_per_class)
    # print('label per class:', label_per_class)
    t_min_class = label_per_class[0]
    t_max_class = label_per_class[-1]

    feature = torch.cat((s_feature, t_feature), 0)
    feature_cent = torch.mean(feature, dim=0)
    label = torch.cat((s_label, t_label))
    label_numpy = label.cpu().detach().numpy()
    # print('label_numpy:', label_numpy)
    class_label = np.unique(label_numpy)
    # print('class_label:',class_label)
    for i in class_label:
        idx = np.argwhere(label_numpy == i).reshape(-1)

        feature_i = feature[idx]

        feature_i_cent = torch.mean(feature_i, dim=0)
        dist1 = torch.norm((feature_i - feature_i_cent), p=2).to(s_label.device)
        dist2 = torch.norm((feature_i_cent - feature_cent), p=2).to(s_label.device)
        numerator += dist1
        denominator += dist2

    loss = numerator / denominator * t_min_class / t_max_class
    return loss


class CenterLoss(nn.Module):
    """Center loss.
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=5, feat_dim=2560, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
            # self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        ##distmat = 1*distmat-2*x*self.centers.t()
        distmat.addmm_(1, -2, x, self.centers.t())
        # print('centers:', self.centers)
        classes = torch.arange(self.num_classes).long()  # torch.int64
        if self.use_gpu:
            classes = classes.cuda()
        labels = labels.long()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        # print(mask)

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        # dist = []
        # for i in range(batch_size):
        #     # print(mask[i])
        #     value = distmat[i][mask[i]]
        #     value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
        #     dist.append(value)
        # dist = torch.cat(dist)
        # loss = dist.mean()
        return loss


class MMD_loss(nn.Module):
    def __init__(self, source_label=None, class_ratio=None, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type
        self.class_ratio = class_ratio
        self.source_label = source_label

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y, source_label, class_ratio):
        loss = 0.0
        source_mean = 0
        weights = []
        for i in range(f_of_X.shape[0]):
            weight = class_ratio[source_label[i]]
            weights.append(weight)
            source_mean += class_ratio[source_label[i]] * f_of_X[i]
        source_mean = source_mean / sum(weights)
        # delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        delta = source_mean - f_of_Y.float().mean(0)
        # loss = delta.dot(delta.T)
        loss = torch.norm(delta) ** 2
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target, self.source_label, self.class_ratio)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
            torch.cuda.empty_cache()
            return loss

