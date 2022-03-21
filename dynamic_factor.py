import numpy as np

#from thundersvm import SVC
from sklearn import svm

import time
#import cupy as cp

def proxy_a_distance(source_X, target_X, verbose=False):
    """
    Compute the Proxy-A-Distance of a source/target representation
    """
    nb_source = np.shape(source_X)[0]
    nb_target = np.shape(target_X)[0]

    if verbose:
        print('PAD on', (nb_source, nb_target), 'examples')

    C_list = np.logspace(-5, 4, 10)

    half_source, half_target = int(nb_source/2), int(nb_target/2)
    train_X = np.vstack((source_X[0:half_source, :], target_X[0:half_target, :]))
    train_Y = np.hstack((np.zeros(half_source, dtype=int), np.ones(half_target, dtype=int)))

    test_X = np.vstack((source_X[half_source:, :], target_X[half_target:, :]))
    test_Y = np.hstack((np.zeros(nb_source - half_source, dtype=int), np.ones(nb_target - half_target, dtype=int)))

    best_risk = 1.0
    start = time.time()
    #500ms
    for C in C_list:

        clf = SVC(C=C, kernel='linear', verbose=False)
        clf.fit(train_X, train_Y)

        train_risk = np.mean(clf.predict(train_X) != train_Y)
        test_risk = np.mean(clf.predict(test_X) != test_Y)

        if verbose:
            print('[ PAD C = %f ] train risk: %f  test risk: %f' % (C, train_risk, test_risk))

        if test_risk > .5:
            test_risk = 1. - test_risk

        best_risk = min(best_risk, test_risk)
    torch.cuda.synchronize()
    end = time.time()
    print('Time:{}ms'.format((end - start) * 1000))
    return 2 * (1. - 2 * best_risk)


def estimate_mu(_X1, _Y1, _X2, _Y2):
    """
    Estimate value of mu using conditional and marginal A-distance.
    """

    adist_m = proxy_a_distance(_X1, _X2)

    Cs, Ct = np.unique(_Y1), np.unique(_Y2)
    #返回Cs Ct的共同元素
    C = np.intersect1d(Cs, Ct)

    epsilon = 1e-3
    list_adist_c = []
    tc = len(C)

    for i in C:
        ind_i, ind_j = np.where(_Y1 == i), np.where(_Y2 == i)
        Xsi = _X1[ind_i[0], :]
        Xtj = _X2[ind_j[0], :]
        if len(Xsi) <= 1 or len(Xtj) <= 1:
            tc -= 1
            continue
        adist_i = proxy_a_distance(Xsi, Xtj)
        list_adist_c.append(adist_i)
    if tc < 1:
        return 0
    adist_c = sum(list_adist_c) / tc
    mu = adist_c / (adist_c + adist_m)
    if mu > 1:
        mu = 1
    if mu < epsilon:
        mu = 0

    return mu


#weight.py

def convert_to_onehot(sca_label, class_num=5):
    return np.eye(class_num)[sca_label]

class Weight:

    @staticmethod
    def cal_weight(s_label, t_label, type='visual', batch_size=32, class_num=5):
        batch_size = s_label.size()[0]
        s_sca_label = s_label.cpu().data.numpy()
        s_vec_label = convert_to_onehot(s_sca_label)
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)
        s_sum[s_sum == 0] = 100
        s_vec_label = s_vec_label / s_sum

        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        #t_vec_label = convert_to_onehot(t_sca_label)

        t_vec_label = t_label.cpu().data.numpy()
        t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
        t_sum[t_sum == 0] = 100
        t_vec_label = t_vec_label / t_sum
        #t_sca_label = [sampels,1]
        #t_vec_label = [samples,class]
        weight_ss = np.zeros((batch_size, batch_size))
        weight_tt = np.zeros((batch_size, batch_size))
        weight_st = np.zeros((batch_size, batch_size))

        set_s = set(s_sca_label)
        set_t = set(t_sca_label)
        count = 0
        for i in range(class_num):
            if i in set_s and i in set_t:
                s_tvec = s_vec_label[:, i].reshape(batch_size, -1)
                t_tvec = t_vec_label[:, i].reshape(batch_size, -1)
                ss = np.dot(s_tvec, s_tvec.T)
                weight_ss = weight_ss + ss# / np.sum(s_tvec) / np.sum(s_tvec)
                tt = np.dot(t_tvec, t_tvec.T)
                weight_tt = weight_tt + tt# / np.sum(t_tvec) / np.sum(t_tvec)
                st = np.dot(s_tvec, t_tvec.T)
                weight_st = weight_st + st# / np.sum(s_tvec) / np.sum(t_tvec)
                count += 1

        length = count  # len( set_s ) * len( set_t )
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')


#mmd.py

import torch
import torch.nn as nn




class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        #input = [samples,features]
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        #
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def marginal(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss

    def conditional(self, source, target, s_label, t_label, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        batch_size = source.size()[0]
        weight_ss, weight_tt, weight_st = Weight.cal_weight(
            s_label, t_label, type='visual')
        weight_ss = torch.from_numpy(weight_ss).cuda()
        weight_tt = torch.from_numpy(weight_tt).cuda()
        weight_st = torch.from_numpy(weight_st).cuda()

        kernels = self.guassian_kernel(source, target,
                                kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        loss = torch.Tensor([0]).cuda()
        if torch.sum(torch.isnan(sum(kernels))):
            return loss
        SS = kernels[:batch_size, :batch_size]
        TT = kernels[batch_size:, batch_size:]
        ST = kernels[:batch_size, batch_size:]

        loss += torch.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)
        return loss
