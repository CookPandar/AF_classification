# -*- coding: utf-8 -*-

import numpy as np
import wfdb

from scipy.signal import resample
import glob
import pywt

import pandas as pd


# -----------------小波多分辨率分解-----------------

def normalize(data):
    data = data.astype('float')
    mx = np.max(data, axis=0).astype(np.float64)
    mn = np.min(data, axis=0).astype(np.float64)
    # Workaround to solve the problem of ZeroDivisionError
    return np.true_divide(data - mn, mx - mn, out=np.zeros_like(data - mn), where=(mx - mn)!=0)

def RemovalOfBaselineDrift(data):
    wavelet = 'db5'
    X = range(len(data))
    w = pywt.Wavelet('db5')  # 选用Daubechies6小波
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)  # level:  分解阶次。使用dwt_max_level（）时，计算信号能达到的最高分解阶次。
    # print("maximum level is " + str(maxlev))
    wave = pywt.wavedec(data, 'db5', level=8)  # 将信号进行小波分解

    # 小波重构
    # ya8 = pywt.waverec(np.multiply(wave, [1, 0, 0, 0, 0, 0, 0, 0, 0]).tolist(), wavelet)
    # yd8 = pywt.waverec(np.multiply(wave, [0, 1, 0, 0, 0, 0, 0, 0, 0]).tolist(), wavelet)
    # yd7 = pywt.waverec(np.multiply(wave, [0, 0, 1, 0, 0, 0, 0, 0, 0]).tolist(), wavelet)
    # yd6 = pywt.waverec(np.multiply(wave, [0, 0, 0, 1, 0, 0, 0, 0, 0]).tolist(), wavelet)
    # yd5 = pywt.waverec(np.multiply(wave, [0, 0, 0, 0, 1, 0, 0, 0, 0]).tolist(), wavelet)
    # yd4 = pywt.waverec(np.multiply(wave, [0, 0, 0, 0, 0, 1, 0, 0, 0]).tolist(), wavelet)
    # yd3 = pywt.waverec(np.multiply(wave, [0, 0, 0, 0, 0, 0, 1, 0, 0]).tolist(), wavelet)
    # yd2 = pywt.waverec(np.multiply(wave, [0, 0, 0, 0, 0, 0, 0, 1, 0]).tolist(), wavelet)
    # yd1 = pywt.waverec(np.multiply(wave, [0, 0, 0, 0, 0, 0, 0, 0, 1]).tolist(), wavelet)
    P = pywt.waverec(np.multiply(wave, [0, 1, 1, 1, 1, 1, 1, 1, 1]).tolist(), wavelet)
    return P


# --------------------小波去噪-----------------
def WTfilt_1d(sig):
    """
    对信号进行小波变换滤波
    :param sig: 输入信号，1-d array
    :return: 小波滤波后的信号，1-d array
    """
    coeffs = pywt.wavedec(sig, 'db6', level=9)
    coeffs[-1] = np.zeros(len(coeffs[-1]))
    coeffs[-2] = np.zeros(len(coeffs[-2]))
    coeffs[0] = np.zeros(len(coeffs[0]))
    sig_filt = pywt.waverec(coeffs, 'db6')
    return sig_filt


# --------------------去噪-----------------
def FilteredECG(sig):
    Record = RemovalOfBaselineDrift(sig)
    sig_filt = WTfilt_1d(Record)
    return sig_filt

# -------------------------心拍截取-------------------
def heartbeat(file0,panth):
    '''
    file0:下载的MITAB数据

    '''
    N_Seg = [];
    SVEB_Seg = [];
    VEB_Seg = [];
    F_Seg = [];
    Q_Seg = [];

    annotation = wfdb.rdann(panth + '/'+file0[-7:-4], 'atr')
    record_name = annotation.record_name  # 读取记录名称
    Record = wfdb.rdsamp(panth+ '/'+record_name)[0][:, 0]  # 一般只取一个导联
    record = FilteredECG(Record)  # 去噪
    label = annotation.symbol  # 心拍标签列表
    label_index = annotation.sample  # 标签索引列表
    for j in range(len(label_index)):
        if label_index[j] >= 144 and (label_index[j] + 180) <= 650000:
            if label[j] == 'N' or label[j] == '.' or label[j] == 'L' or label[j] == 'R' or label[j] == 'e' or label[j] == 'j':
                Seg = record[label_index[j] - 144:label_index[j] + 180]  # R峰的前0.4s和后0.5s
                segment = resample(Seg, 251, axis=0)  # 重采样到251
                N_Seg.append(segment)

            if label[j] == 'A' or label[j] == 'a' or label[j] == 'J' or label[j] == 'S':
                Seg = record[label_index[j] - 144:label_index[j] + 180]
                segment = resample(Seg, 251, axis=0)
                SVEB_Seg.append(segment)

            if label[j] == 'V' or label[j] == 'E':
                Seg = record[label_index[j] - 144:label_index[j] + 180]
                segment = resample(Seg, 251, axis=0)
                VEB_Seg.append(segment)

            if label[j] == 'F':
                Seg = record[label_index[j] - 144:label_index[j + 1] + 180]
                segment = resample(Seg, 251, axis=0)
                F_Seg.append(segment)
            if label[j] == '/' or label[j] == 'f' or label[j] == 'Q':
                Seg = record[label_index[j] - 144:label_index[j] + 180]
                segment = resample(Seg, 251, axis=0)
                Q_Seg.append(segment)

    N_segement = np.array(N_Seg)
    SVEB_segement = np.array(SVEB_Seg)
    VEB_segement = np.array(VEB_Seg)
    F_segement = np.array(F_Seg)
    Q_segement = np.array(Q_Seg)

    label_N = np.zeros(N_segement.shape[0])
    label_SVEB = np.ones(SVEB_segement.shape[0])
    label_VEB = np.ones(VEB_segement.shape[0]) * 2
    label_F = np.ones(F_segement.shape[0]) * 3
    label_Q = np.ones(Q_segement.shape[0]) * 4

    Data = N_segement
    Label = label_N
    if SVEB_segement.size!=0 :
        Data = np.concatenate((Data, SVEB_segement), axis=0)
        Label = np.concatenate((Label, label_SVEB), axis=0)
    if  VEB_segement.size!=0:
        Data = np.concatenate((Data, VEB_segement), axis=0)
        Label = np.concatenate((Label, label_VEB), axis=0)
    if F_segement.size != 0:
        Data = np.concatenate((Data,F_segement), axis=0)
        Label = np.concatenate((Label, label_F), axis=0)
    if Q_segement.size != 0:
        Data = np.concatenate((Data,Q_segement), axis=0)
        Label = np.concatenate((Label, label_Q), axis=0)

    return Data, Label




def heartbeat_uda(file0,panth):
    '''
    file0:下载的MITAB数据

    '''
    N_Seg = [];
    SVEB_Seg = [];
    VEB_Seg = [];
    F_Seg = [];
    Q_Seg = [];

    N_Seg_pre = [];
    SVEB_Seg_pre = [];
    VEB_Seg_pre = [];
    F_Seg_pre = [];
    Q_Seg_pre = [];

    annotation = wfdb.rdann(panth + '/'+file0[-7:-4], 'atr')
    record_name = annotation.record_name  # 读取记录名称
    Record = wfdb.rdsamp(panth+ '/'+record_name)[0][:, 0]  # 一般只取一个导联
    record = FilteredECG(Record)  # 去噪
    label = annotation.symbol  # 心拍标签列表
    label_index = annotation.sample  # 标签索引列表
    for j in range(len(label_index)):
        if label_index[j] >= 144 and (label_index[j] + 180) <= 650000:
            #前五分钟，每秒采样361，则为361*60*5=108300前的数据
            if label_index[j]<=108300:
                if label[j] == 'N' or label[j] == '.' or label[j] == 'L' or label[j] == 'R' or label[j] == 'e' or label[
                    j] == 'j':
                    Seg = record[label_index[j] - 144:label_index[j] + 180]  # R峰的前0.4s和后0.5s
                    segment = resample(Seg, 251, axis=0)  # 重采样到251
                    N_Seg_pre.append(segment)

                if label[j] == 'A' or label[j] == 'a' or label[j] == 'J' or label[j] == 'S':
                    Seg = record[label_index[j] - 144:label_index[j] + 180]
                    segment = resample(Seg, 251, axis=0)
                    SVEB_Seg_pre.append(segment)

                if label[j] == 'V' or label[j] == 'E':
                    Seg = record[label_index[j] - 144:label_index[j] + 180]
                    segment = resample(Seg, 251, axis=0)
                    VEB_Seg_pre.append(segment)

                if label[j] == 'F':
                    Seg = record[label_index[j] - 144:label_index[j + 1] + 180]
                    segment = resample(Seg, 251, axis=0)
                    F_Seg_pre.append(segment)
                if label[j] == '/' or label[j] == 'f' or label[j] == 'Q':
                    Seg = record[label_index[j] - 144:label_index[j] + 180]
                    segment = resample(Seg, 251, axis=0)
                    Q_Seg_pre.append(segment)
            else:
                if label[j] == 'N' or label[j] == '.' or label[j] == 'L' or label[j] == 'R' or label[j] == 'e' or label[j] == 'j':
                    Seg = record[label_index[j] - 144:label_index[j] + 180]  # R峰的前0.4s和后0.5s
                    segment = resample(Seg, 251, axis=0)  # 重采样到251
                    N_Seg.append(segment)

                if label[j] == 'A' or label[j] == 'a' or label[j] == 'J' or label[j] == 'S':
                    Seg = record[label_index[j] - 144:label_index[j] + 180]
                    segment = resample(Seg, 251, axis=0)
                    SVEB_Seg.append(segment)

                if label[j] == 'V' or label[j] == 'E':
                    Seg = record[label_index[j] - 144:label_index[j] + 180]
                    segment = resample(Seg, 251, axis=0)
                    VEB_Seg.append(segment)

                if label[j] == 'F':
                    Seg = record[label_index[j] - 144:label_index[j + 1] + 180]
                    segment = resample(Seg, 251, axis=0)
                    F_Seg.append(segment)
                if label[j] == '/' or label[j] == 'f' or label[j] == 'Q':
                    Seg = record[label_index[j] - 144:label_index[j] + 180]
                    segment = resample(Seg, 251, axis=0)
                    Q_Seg.append(segment)
    #测试数据
    N_segement = np.array(N_Seg)
    SVEB_segement = np.array(SVEB_Seg)
    VEB_segement = np.array(VEB_Seg)
    F_segement = np.array(F_Seg)
    Q_segement = np.array(Q_Seg)

    label_N = np.zeros(N_segement.shape[0])
    label_SVEB = np.ones(SVEB_segement.shape[0])
    label_VEB = np.ones(VEB_segement.shape[0]) * 2
    label_F = np.ones(F_segement.shape[0]) * 3
    label_Q = np.ones(Q_segement.shape[0]) * 4

    Data = N_segement
    Label = label_N
    if SVEB_segement.size!=0 :
        Data = np.concatenate((Data, SVEB_segement), axis=0)
        Label = np.concatenate((Label, label_SVEB), axis=0)
    if  VEB_segement.size!=0:
        Data = np.concatenate((Data, VEB_segement), axis=0)
        Label = np.concatenate((Label, label_VEB), axis=0)
    if F_segement.size != 0:
        Data = np.concatenate((Data,F_segement), axis=0)
        Label = np.concatenate((Label, label_F), axis=0)
    if Q_segement.size != 0:
        Data = np.concatenate((Data,Q_segement), axis=0)
        Label = np.concatenate((Label, label_Q), axis=0)

    #训练数据

    N_segement_pre = np.array(N_Seg_pre)
    SVEB_segement_pre = np.array(SVEB_Seg_pre)
    VEB_segement_pre = np.array(VEB_Seg_pre)
    F_segement_pre = np.array(F_Seg_pre)
    Q_segement_pre = np.array(Q_Seg_pre)
    label_N_pre = np.zeros(N_segement.shape[0])
    label_SVEB_pre = np.ones(SVEB_segement.shape[0])
    label_VEB_pre = np.ones(VEB_segement.shape[0]) * 2
    label_F_pre = np.ones(F_segement.shape[0]) * 3
    label_Q_pre = np.ones(Q_segement.shape[0]) * 4
    train_data = N_segement_pre
    train_label = label_N_pre

    if SVEB_segement_pre.size!=0 :
        Data = np.concatenate((train_data, SVEB_segement_pre), axis=0)
        Label = np.concatenate((train_label, label_SVEB_pre), axis=0)
    if  VEB_segement_pre.size!=0:
        Data = np.concatenate((train_data, VEB_segement_pre), axis=0)
        Label = np.concatenate((train_label, label_VEB_pre), axis=0)
    if F_segement_pre.size != 0:
        Data = np.concatenate((train_data,F_segement_pre), axis=0)
        Label = np.concatenate((train_label, label_F_pre), axis=0)
    if Q_segement_pre.size != 0:
        Data = np.concatenate((train_data,Q_segement_pre), axis=0)
        Label = np.concatenate((train_label, label_Q_pre), axis=0)

    return Data, Label,train_data,train_label






