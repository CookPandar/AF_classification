# encoding: utf-8
import wfdb #wfdb==1.2.2
import numpy as np
import scipy.io as sio
import os
import json
import pickle
import random
np.set_printoptions(suppress = True)
from scipy import signal
from scipy.signal import butter, lfilter, freqz
from scipy.signal import medfilt
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import shutil


def wtmedian_denoise(sig, gain_mask=[0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], baseline0_windows=36, baseline1_windows=108):
    import dtcwt
    from scipy import signal
    from scipy.signal import butter, lfilter, freqz
    from scipy.signal import medfilt

    baseline0 = medfilt(sig, baseline0_windows * 2 + 1)
    baseline1 = medfilt(baseline0, baseline1_windows * 2 + 1)
    sig_denoised = sig - baseline1
    transform = dtcwt.Transform1d()
    #sig是列向量
    sig_t = transform.forward(sig_denoised, nlevels=len(gain_mask))
    # sig_t = transform.forward(sig, nlevels=len(gain_mask))
    sig_recon = transform.inverse(sig_t, gain_mask)
    #200ms和600ms的中值滤波（采样率位360Hz)
    # baseline0 = medfilt(sig_recon, baseline0_windows*2+1)
    # baseline1 = medfilt(baseline0, baseline1_windows*2+1)
    # sig_denoised = sig_recon - baseline1
    # baseline0 = [np.median(sig_recon[max(0, x - 36):min(x + 36, len(sig_recon) - 1)])
    #              for x in range(len(sig_recon))]
    # baseline1 = [np.median(baseline0[max(
    #     0, x - 108):min(x + 108, len(baseline0) - 1)]) for x in range(len(baseline0))]
    # sig_denoised = list(map((lambda x, y: x - y), sig_recon, baseline1))
    return sig_recon

#To do:
def SDL_SR_denoise(sig):
    return sig


#专门准备DS1训练数据，将所有训练数据都分成一群片段的  DS1\train
def mitdb_data_prepare_single_128samples_beat(data_dir, data_list, save_dir):
    '''参考论文：Automated Heartbeat Classification Using 3-D Inputs Based on Convolutional Neural
                Network With Multi-Fields of View
       心跳分段：将mitdb导联II的数据单个beat(从前1个R峰后的0.14s处开始,到当前R峰之后的0.28s结束,采样率位360Hz)
       得到的每个beat长度不等，因而resample到长度为M=128的beat

       每个样本由current heartbeat、pre_RR_tatio、near_pre_RR_ratio组成，shape(3,128,1)
       '''
    folder_name = save_dir.split('/')[2] + '/'
    Heartbeats_img = 'exp1/HeartBeatsImg/' + folder_name
    if os.path.exists(Heartbeats_img):
        shutil.rmtree(Heartbeats_img)
    if not os.path.exists(Heartbeats_img):
        os.makedirs(Heartbeats_img)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fs = 360#257、128
    left_beat = int(fs * 0.14)#0.14
    right_beat = int(fs * 0.28)#0.28
    hrv_length = 5 - 1
    feature_num = 3
    channel = 1
    # left_beat = 70
    # right_beat = 99
    beat_length = 128
    '''not AAMI standard, but most papers like this'''
    #标签重编码
    label_t5 = {'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
                'A': 'S', 'a':'S', 'J':'S', 'S': 'S',
                'V': 'V', 'E': 'V',
                'F': 'F',
                '/': 'Q', 'f':'Q', 'Q':'Q'
                }

    num = 0
    for cur_record_id in data_list:
        print(cur_record_id)
        record_info = wfdb.rdsamp(os.path.join(data_dir, str(cur_record_id)))
        record_ann = wfdb.rdann(os.path.join(data_dir, str(cur_record_id)), 'atr')

        samples = record_ann.sample
        symbols = record_ann.symbol

        MLII_index = record_info[1]['sig_name'].index('MLII')#'MLII'、'II'、'ECG1'
        # sig = record_info.p_signals[:,0:1]
        sig = record_info[0][:, MLII_index]
        sig = sig.reshape(1, len(sig))

        # sig = wtmedian_denoise(sig[0], gain_mask = [0, 1, 1, 1, 1, 1, 1, 1, 1, 0 , 0])
        # sig = sig.reshape(1, len(sig))

        # sig = (sig - np.mean(sig)) / np.std(sig)
        sig_length = sig.shape[1]
        print('sig shape:', sig.shape)


        ECG_Data = np.array([]).reshape((0, feature_num, beat_length, channel))
        # ECG_Data = np.array([]).reshape((0, beat_length))
        # RR_feature = np.array([]).reshape((0, hrv_length))
        # print(ECG_Data.shape)
        Label = np.array([]).reshape((0, 5))

        beat_label = []
        r_position = []
        for i, symbol in enumerate(symbols):
            #对不属于这16类的心跳，在计算平均RR间期时也需要考虑，不能跳过这些心跳
            r_position.append(samples[i])
            if symbol in label_t5.keys():
                beat_label.append(symbol)
        rr_interval = np.diff(r_position)
        mean_rr = np.mean(rr_interval)
        # rr_interval = (rr_interval - np.min(rr_interval))/(np.max(rr_interval) - np.min(rr_interval ))
        # rr_interval = rr_interval / mean_rr
        print('len beat_label:',len(beat_label))
        # print('mean_rr:',mean_rr )
        # left_beat = int(mean_rr * 0.6)  # 0.14
        # right_beat = int(mean_rr * 0.3)  # 0.28

        # for i, label in enumerate(beat_label):
        for i, symbol in enumerate(symbols):
            '''这样做还是存在相邻的心跳不属于这15类、但划分的心跳包含部分这些不在15类中的心跳信息的可能'''
            if i >= 2 and i <= len(symbols)-1 and symbol in label_t5.keys():
            # if i >= 1  and symbol in label_t5.keys():
                #截取两个心跳的操作
                if i == 2:
                    ##注意：第1个标注samples[0]并非R峰
                    # one_beat = sig[0, :samples[i]+ right_beat + 1]
                    one_beat = sig[0, samples[i - 1] + left_beat:samples[i] + right_beat + 1]
                    one_beat = one_beat - np.mean(one_beat)
                    one_beat = signal.resample(one_beat, beat_length)
                    one_beat = one_beat.reshape(1, one_beat.shape[0])
                    pre_rr_ratio = rr_interval[i - 1] / np.mean(rr_interval[:i])
                    pre_rr_ratio = np.tile(pre_rr_ratio, (1, beat_length))
                    near_pre_rr_ratio = pre_rr_ratio


                if samples[i - 1] + left_beat < samples[i] and samples[i] + right_beat + 1 < sig_length:
                     one_beat = sig[0, samples[i - 1] + left_beat:samples[i] + right_beat + 1]
                     one_beat = one_beat - np.mean(one_beat)
                     one_beat = signal.resample(one_beat, beat_length)
                     one_beat = one_beat.reshape(1, one_beat.shape[0])
                     pre_rr_ratio = rr_interval[i - 1] / np.mean(rr_interval[:i])
                     pre_rr_ratio = np.tile(pre_rr_ratio, (1, beat_length))
                     if i <= 12:
                         near_pre_rr_ratio = pre_rr_ratio
                     else:
                         near_pre_rr_ratio = rr_interval[i - 1] / np.mean(rr_interval[i - 10:i])
                         near_pre_rr_ratio = np.tile(near_pre_rr_ratio, (1, beat_length))


                if num == 0 and i <= 11:
                    '''保存第一个record的前10个Beats'''
                    plt.figure()
                    plt.plot(one_beat[0])
                    plt.grid(True)
                    plt.savefig(Heartbeats_img + str(cur_record_id) + '_' + str(i) + '.png')

                ecg_data = np.concatenate((one_beat, pre_rr_ratio, near_pre_rr_ratio),axis=0)
                ecg_data = ecg_data.reshape(1, feature_num, beat_length, channel)
                ECG_Data = np.concatenate((ECG_Data, ecg_data), axis=0)
                # RR_feature = np.concatenate((RR_feature, rr_feature), axis=0)
                label = symbol
                if label_t5[label] == 'N':
                    cur_label = np.array([[1, 0, 0, 0, 0]])
                elif label_t5[label]  == 'S':
                    cur_label = np.array([[0, 1, 0, 0, 0]])
                elif label_t5[label]  == 'V':
                    cur_label = np.array([[0, 0, 1, 0, 0]])
                elif label_t5[label]  == 'F':
                    cur_label = np.array([[0, 0, 0, 1, 0]])
                elif label_t5[label] == 'Q':
                    cur_label = np.array([[0, 0, 0, 0, 1]])
                Label = np.concatenate((Label, cur_label), axis=0)

        print(ECG_Data.shape)
        # print(RR_feature.shape)
        # print(RR_feature[0])
        print(Label.shape)


        # np.savez(save_dir+str(cur_record_id), ECG_Data=ECG_Data, RR_feature=RR_feature, Label=Label)
        np.savez(save_dir + str(cur_record_id), ECG_Data=ECG_Data, Label=Label)
        num = num + 1
        # # if num == 1:
        # #     break

        print('=====Processing第'+str(num)+'个record=====\n')

#专门准备DS2测试数据和训练数据，前5min数据为DS2\train   之后的数据为DS2\test
def mitdb_data_prepare_single_128samples_beat_for_UDA(data_dir, data_list, save_dir):
    '''参考论文：Automated Heartbeat Classification Using 3-D Inputs Based on Convolutional Neural
                Network With Multi-Fields of View
       心跳分段：将mitdb导联II的数据单个beat(从前1个R峰后的0.14s处开始,到当前R峰之后的0.28s结束,采样率位360Hz)
       得到的每个beat长度不等，因而resample到长度为M=128的beat


       每个样本由current heartbeat、pre_RR_tatio、near_pre_RR_ratio组成，shape(3,128,1)
       '''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    DS2_train_save_dir = save_dir + '/DS2_train/'
    DS2_test_save_dir = save_dir + '/DS2_test/'
    if not os.path.exists(DS2_train_save_dir):
        os.makedirs(DS2_train_save_dir)
    if not os.path.exists(DS2_test_save_dir):
        os.makedirs(DS2_test_save_dir)

    folder_name = save_dir.split('/')[2] + '/'
    Heartbeats_img = '../exp1/HeartBeatsImg/' + folder_name
    if os.path.exists(Heartbeats_img):
        shutil.rmtree(Heartbeats_img)
    if not os.path.exists(Heartbeats_img):
        os.makedirs(Heartbeats_img)


    fs = 360
    left_beat = int(fs * 0.14)#0.14
    right_beat = int(fs * 0.28)#0.28
    hrv_length = 5 - 1
    feature_num = 3
    channel = 1

    # left_beat = 70
    # right_beat = 99
    beat_length = 128

    label_t5 = {'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
                'A': 'S', 'a':'S', 'J':'S', 'S': 'S',
                'V': 'V', 'E': 'V',
                'F': 'F',
                '/': 'Q', 'f':'Q', 'Q':'Q'
                }

    num = 0
    for cur_record_id in data_list:
        print(cur_record_id)
        record_info = wfdb.rdsamp(os.path.join(data_dir, str(cur_record_id)))
        record_ann = wfdb.rdann(os.path.join(data_dir, str(cur_record_id)), 'atr')
        samples = record_ann.sample
        symbols = record_ann.symbol

        MLII_index = record_info[1]['sig_name'].index('MLII')
        # MLII_index = record_info.signame.index('ECG1')

        # MLII_index = record_info.signame.index('II')
        sig = record_info[0][:, MLII_index]
        origin_sig_length = len(sig)
        print('原始sig长度:', origin_sig_length)

        # sig = signal.resample(sig, int(360/257*origin_sig_length))
        sig = sig.reshape(1, len(sig))


        # sig = wtmedian_denoise(sig[0], gain_mask = [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
        # sig = sig.reshape(1, len(sig))
        # sig = (sig - np.mean(sig)) / np.std(sig)
        sig_length = sig.shape[1]
        print('sig shape:',sig.shape)
        # print('采样后的sig长度：',sig_length )


        ECG_Data_train = np.array([]).reshape((0, feature_num, beat_length, channel))
        ECG_Data_test = np.array([]).reshape((0, feature_num, beat_length, channel))
        # ECG_Data = np.array([]).reshape((0, beat_length))
        # RR_feature = np.array([]).reshape((0, hrv_length))
        # print(ECG_Data.shape)
        Label_train = np.array([]).reshape((0, 5))
        Label_test = np.array([]).reshape((0, 5))

        beat_label = []
        r_position = []
        for i, symbol in enumerate(symbols):
            #对不属于这16类的心跳，在计算平均RR间期时也需要考虑，不能跳过这些心跳
            r_position.append(int(samples[i]*360/360))
            if symbol in label_t5.keys():
                beat_label.append(symbol)

        rr_interval = np.diff(r_position)
        mean_rr = np.mean(rr_interval)
        # rr_interval = (rr_interval - np.min(rr_interval))/(np.max(rr_interval) - np.min(rr_interval ))
        # rr_interval = rr_interval / mean_rr
        print('len beat_label:',len(beat_label))
        print('mean_rr:',mean_rr )
        # left_beat = int(mean_rr * 0.6)  # 0.14
        # right_beat = int(mean_rr * 0.3)  # 0.28

        # for i, label in enumerate(beat_label):
        for i, symbol in enumerate(symbols):
            '''这样做还是存在相邻的心跳不属于这15类、但划分的心跳包含部分这些不在15类中的心跳信息的可能'''

            if i >= 2 and i <= len(symbols)-1 and symbol in label_t5.keys():
            # if i >= 1  and symbol in label_t5.keys():
                #截取两个心跳的操作

                if i == 2:
                    ##注意：第1个标注samples[0]并非R峰
                    one_beat = sig[0, int(samples[i-1]*360/360) + left_beat:int(samples[i]*360/360)+ right_beat + 1]

                    one_beat = one_beat - np.mean(one_beat)
                    one_beat = signal.resample(one_beat, beat_length)

                    one_beat = one_beat.reshape(1, one_beat.shape[0])
                    pre_rr_ratio = rr_interval[i - 1] / np.mean(rr_interval[:i])
                    pre_rr_ratio = np.tile(pre_rr_ratio, (1, beat_length))
                    near_pre_rr_ratio = pre_rr_ratio

                if int(samples[i-1]*360/360)+ left_beat < int(samples[i]*360/360) and int(samples[i]*360/360)\
                        + right_beat + 1 < sig_length:

                     one_beat = sig[0, int(samples[i-1]*360/360) + left_beat:int(samples[i]*360/360) + right_beat + 1]

                     one_beat = one_beat - np.mean(one_beat)
                     one_beat = signal.resample(one_beat, beat_length)
                     one_beat = one_beat.reshape(1, one_beat.shape[0])
                     pre_rr_ratio = rr_interval[i - 1] / np.mean(rr_interval[:i])
                     pre_rr_ratio = np.tile(pre_rr_ratio, (1, beat_length))
                     if i <= 12:
                         near_pre_rr_ratio = pre_rr_ratio
                     else:
                         near_pre_rr_ratio = rr_interval[i - 1] / np.mean(rr_interval[i - 10:i])
                         near_pre_rr_ratio = np.tile(near_pre_rr_ratio, (1, beat_length))

                if num == 0 and i <= 11 :
                    '''保存第一个record的前10个Beats'''
                    plt.figure()
                    plt.plot(one_beat[0])
                    plt.grid(True)
                    plt.savefig(Heartbeats_img + str(cur_record_id) + '_' + str(i) + '.png')

                ecg_data = np.concatenate((one_beat, pre_rr_ratio, near_pre_rr_ratio), axis=0)
                ecg_data = ecg_data.reshape(1, feature_num, beat_length, channel)
                # RR_feature = np.concatenate((RR_feature, rr_feature), axis=0)
                label = symbol
                if label_t5[label] == 'N':
                    cur_label = np.array([[1, 0, 0, 0, 0]])
                elif label_t5[label] == 'S':
                    cur_label = np.array([[0, 1, 0, 0, 0]])
                elif label_t5[label] == 'V':
                    cur_label = np.array([[0, 0, 1, 0, 0]])
                elif label_t5[label] == 'F':
                    cur_label = np.array([[0, 0, 0, 1, 0]])
                elif label_t5[label] == 'Q':
                    cur_label = np.array([[0, 0, 0, 0, 1]])


                '''前5min数据'''
                if samples[i] <= fs*5*60:
                    ECG_Data_train = np.concatenate((ECG_Data_train, ecg_data), axis=0)
                    Label_train = np.concatenate((Label_train, cur_label), axis=0)
                else:
                    ECG_Data_test = np.concatenate((ECG_Data_test, ecg_data), axis=0)
                    Label_test = np.concatenate((Label_test, cur_label), axis=0)


        print(ECG_Data_train.shape)
        print(Label_train.shape)
        print(ECG_Data_test.shape)
        print(Label_test.shape)

        # np.savez(save_dir+str(cur_record_id), ECG_Data=ECG_Data, RR_feature=RR_feature, Label=Label)
        np.savez(DS2_train_save_dir + str(cur_record_id), ECG_Data=ECG_Data_train, Label=Label_train)
        np.savez(DS2_test_save_dir + str(cur_record_id), ECG_Data=ECG_Data_test, Label=Label_test)
        num = num + 1
        # # if num == 1:
        # #     break

        print('=====Processing第'+str(num)+'个record=====\n')

#专门准备DS2测试数据和训练数据，前5min数据为SVDB\train   之后的数据为SVDB\test
def svdb_data_prepare_single_128samples_beat_for_UDA(data_dir, data_list, save_dir):
    '''参考论文：Automated Heartbeat Classification Using 3-D Inputs Based on Convolutional Neural
                Network With Multi-Fields of View
       心跳分段：将mitdb导联II的数据单个beat(从前1个R峰后的0.14s处开始,到当前R峰之后的0.28s结束,采样率位360Hz)
       得到的每个beat长度不等，因而resample到长度为M=128的beat


       每个样本由current heartbeat、pre_RR_tatio、near_pre_RR_ratio组成，shape(3,128,1)
       '''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    DS2_train_save_dir = save_dir + '/train/'
    DS2_test_save_dir = save_dir + '/test/'
    folder_name = save_dir.split('/')[2] + '/'
    Heartbeats_img = '../exp1/HeartBeatsImg/' + folder_name
    if os.path.exists(Heartbeats_img):
        shutil.rmtree(Heartbeats_img)

    if not os.path.exists(DS2_train_save_dir):
        os.makedirs(DS2_train_save_dir)
    if not os.path.exists(DS2_test_save_dir):
        os.makedirs(DS2_test_save_dir)
    if not os.path.exists(Heartbeats_img):
        os.makedirs(Heartbeats_img)

    fs = 128
    left_beat = int(fs * 0.14)  # 0.14
    right_beat = int(fs * 0.28)  # 0.28
    hrv_length = 5 - 1
    feature_num = 3
    channel = 1

    beat_length = 128

    label_t5 = {'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N', '.': 'N',
                'A': 'S', 'a':'S', 'J':'S', 'S': 'S',
                'V': 'V', 'E': 'V',
                'F': 'F',
                '/': 'Q', 'f':'Q', 'Q':'Q'
                }

    num = 0
    for cur_record_id in data_list:
        print(cur_record_id)
        record_info = wfdb.rdsamp(os.path.join(data_dir, str(cur_record_id)))
        # print(record_info.__dict__)
        #读取注释文件
        #record_info 包含了各个通道的数据，并有一个字典表示记载各种描述
        record_ann = wfdb.rdann(os.path.join(data_dir, str(cur_record_id)), 'atr')
        samples = record_ann.sample
        symbols = record_ann.symbol
        #samples 记录每个心拍开始的位置
        #symbols 记录每个心拍的标签
        #print(samples)
        #print(symbols)
        #print(record_info)
        # MLII_index = record_info.signame.index('MLII')##mitdb
        # MLII_index = record_info.signame.index('MLIII')##edb
        # MLII_index = record_info.signame.index('ECG')##stdb
        MLII_index = record_info[1]['sig_name'].index('ECG1')##svdb
        # MLII_index = record_info.signame.index('II')##incartdb
        #print('record_info.p_signals:', record_info[1]['p_signals'].shape)
        sig = record_info[0][:, MLII_index]
        origin_sig_length = len(sig)
        print('原始sig长度:', origin_sig_length)

        # sig = signal.resample(sig, int(360 / fs * origin_sig_length))
        # sig = sig.reshape(1, len(sig))
        #
        # sig = wtmedian_denoise(sig[0], gain_mask=[0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], baseline0_windows=int(0.1 * fs),
        #                        baseline1_windows=int(0.3 * fs))
        sig = sig.reshape(1, len(sig))
        # sig = (sig - np.mean(sig)) / np.std(sig)
        sig_length = sig.shape[1]
        print('sig shape:', sig.shape)
        print('采样后的sig长度：',sig_length )

        #只采用128个信号，
        ECG_Data_train = np.array([]).reshape((0, feature_num, beat_length, channel))
        ECG_Data_test = np.array([]).reshape((0, feature_num, beat_length, channel))
        # ECG_Data = np.array([]).reshape((0, beat_length))
        # RR_feature = np.array([]).reshape((0, hrv_length))
        # print(ECG_Data.shape)
        Label_train = np.array([]).reshape((0, 5))
        Label_test = np.array([]).reshape((0, 5))

        beat_label = []
        r_position = []
        for i, symbol in enumerate(symbols):
            # 对不属于这16类的心跳，在计算平均RR间期时也需要考虑，不能跳过这些心跳
            # r_position.append(int(samples[i] * 360 / fs))
            r_position.append(samples[i])
            if symbol in label_t5.keys():
                beat_label.append(symbol)

        #rr_interval为 rr间期序列
        #r_position为
        rr_interval = np.diff(r_position)
        mean_rr = np.mean(rr_interval)
        # rr_interval = (rr_interval - np.min(rr_interval))/(np.max(rr_interval) - np.min(rr_interval ))
        # rr_interval = rr_interval / mean_rr
        #5类之一的心拍个数
        print('len beat_label:', len(beat_label))
        print('mean_rr:', mean_rr)
        # left_beat = int(mean_rr * 0.6)  # 0.14
        # right_beat = int(mean_rr * 0.3)  # 0.28

        # for i, label in enumerate(beat_label):
        for i, symbol in enumerate(symbols):
            '''这样做还是存在相邻的心跳不属于这15类、但划分的心跳包含部分这些不在15类中的心跳信息的可能'''

            if i >= 2 and i <= len(symbols)-1 and symbol in label_t5.keys():
            # if i >= 1  and symbol in label_t5.keys():
                #截取两个心跳的操作

                if i == 2:
                    ##注意：第1个标注samples[0]并非R峰
                    # one_beat = sig[0, int(samples[i-1]*360/fs) + left_beat:int(samples[i]*360/fs)+ right_beat + 1]
                    one_beat = sig[0, samples[i - 1] + left_beat:samples[i]+ right_beat + 1]
                    one_beat = one_beat - np.mean(one_beat)
                    one_beat = signal.resample(one_beat, beat_length)

                    one_beat = one_beat.reshape(1, one_beat.shape[0])
                    pre_rr_ratio = rr_interval[i - 1] / np.mean(rr_interval[:i])
                    pre_rr_ratio = np.tile(pre_rr_ratio, (1, beat_length))
                    near_pre_rr_ratio = pre_rr_ratio

                # if int(samples[i-1]*360/fs)+ left_beat < int(samples[i]*360/fs) and int(samples[i]*360/fs)\
                #         + right_beat + 1 < sig_length:
                if samples[i - 1] + left_beat < samples[i] and samples[i] + right_beat + 1 < sig_length:

                     # one_beat = sig[0, int(samples[i-1]*360/fs) + left_beat:int(samples[i]*360/fs) + right_beat + 1]
                     one_beat = sig[0, samples[i - 1] + left_beat:samples[i] + right_beat + 1]

                     one_beat = one_beat - np.mean(one_beat)
                     one_beat = signal.resample(one_beat, beat_length)
                     one_beat = one_beat.reshape(1, one_beat.shape[0])
                     pre_rr_ratio = rr_interval[i - 1] / np.mean(rr_interval[:i])
                     pre_rr_ratio = np.tile(pre_rr_ratio, (1, beat_length))
                     if i <= 12:
                         near_pre_rr_ratio = pre_rr_ratio
                     else:
                         near_pre_rr_ratio = rr_interval[i - 1] / np.mean(rr_interval[i - 10:i])
                         near_pre_rr_ratio = np.tile(near_pre_rr_ratio, (1, beat_length))

                if num == 0 and i <= 11 :
                    '''保存第一个record的前10个Beats'''
                    plt.figure()
                    plt.plot(one_beat[0])
                    plt.grid(True)
                    plt.savefig(Heartbeats_img + str(cur_record_id) + '_' + str(i) + '.png')

                ecg_data = np.concatenate((one_beat, pre_rr_ratio, near_pre_rr_ratio), axis=0)
                ecg_data = ecg_data.reshape(1, feature_num, beat_length, channel)
                # RR_feature = np.concatenate((RR_feature, rr_feature), axis=0)
                label = symbol
                if label_t5[label] == 'N':
                    cur_label = np.array([[1, 0, 0, 0, 0]])
                elif label_t5[label] == 'S':
                    cur_label = np.array([[0, 1, 0, 0, 0]])
                elif label_t5[label] == 'V':
                    cur_label = np.array([[0, 0, 1, 0, 0]])
                elif label_t5[label] == 'F':
                    cur_label = np.array([[0, 0, 0, 1, 0]])
                elif label_t5[label] == 'Q':
                    cur_label = np.array([[0, 0, 0, 0, 1]])


                '''前5min数据'''
                # if int(samples[i]*360/fs) <= int(360 * 5 * 60):
                if samples[i] <= fs * 5 * 60:
                    ECG_Data_train = np.concatenate((ECG_Data_train, ecg_data), axis=0)
                    Label_train = np.concatenate((Label_train, cur_label), axis=0)
                else:
                    ECG_Data_test = np.concatenate((ECG_Data_test, ecg_data), axis=0)
                    Label_test = np.concatenate((Label_test, cur_label), axis=0)


        print(ECG_Data_train.shape)
        print(Label_train.shape)
        print(ECG_Data_test.shape)
        print(Label_test.shape)



        # np.savez(save_dir+str(cur_record_id), ECG_Data=ECG_Data, RR_feature=RR_feature, Label=Label)
        np.savez(DS2_train_save_dir + str(cur_record_id), ECG_Data=ECG_Data_train, Label=Label_train)
        np.savez(DS2_test_save_dir + str(cur_record_id), ECG_Data=ECG_Data_test, Label=Label_test)
        num = num + 1
        # # if num == 1:
        # #     break

        print('=====Processing第'+str(num)+'个record=====\n')

def main():
    # '''Prepare exp data: DS1->DS2'''
    ##data_dir存放的文件包括xxx.atr, xxx.data, xxx.hea...
    data_dir = "./MIT-BIH"
    train_record_list = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203,
                         205, 207, 208, 209, 215, 220, 223, 230]
    test_record_list = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214,
                        219, 221, 222, 228, 231, 232, 233, 234]
    # 将读取出的每个reocrd的data和label保存为npz文件
    train_save_dir = 'data/mitdb_uda_DS1/'
    test_save_dir = 'data/mitdb_uda_DS2/'

    mitdb_data_prepare_single_128samples_beat(data_dir, train_record_list, train_save_dir)
    mitdb_data_prepare_single_128samples_beat_for_UDA(data_dir, test_record_list, test_save_dir)



    '''Prepare exp data: MITDB->SVDB'''
    data_dir = "./MIT-BIH"
    train_save_dir = 'data/mitdb_DS1_DS2_whole_data/'
    records_list = train_record_list + test_record_list
    mitdb_data_prepare_single_128samples_beat(data_dir, records_list, train_save_dir)

    # data_dir = "./SVDB"
    # test_save_dir = 'data/svdb_uda_128hz/'
    # data_list = os.listdir(data_dir)
    # data_list = [k for k in data_list if k.startswith('8')]##svdb
    # data_list = np.unique([k[:3] for k in data_list ])
    # print('The number of svdb records:', len(data_list))
    # print(data_list)
    # svdb_data_prepare_single_128samples_beat_for_UDA(data_dir, data_list, test_save_dir)


if __name__ == "__main__":
    main()
