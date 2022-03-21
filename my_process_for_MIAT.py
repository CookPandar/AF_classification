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
from xiaobo import heartbeat
from xiaobo import heartbeat_uda
import torch
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

    feature_num = 1
    channel = 1
    # left_beat = 70
    # right_beat = 99
    beat_length = 251
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
        file_name = os.path.join(data_dir, str(cur_record_id))+'.hea'
        ECG_Data = np.array([]).reshape((0, feature_num, beat_length, channel))

        Label = np.array([]).reshape((0, 5))
        one_beat, cur_label = heartbeat(file_name,data_dir)

        ecg_data = np.concatenate((one_beat),axis=0)
        ecg_data = ecg_data.reshape(-1, feature_num, beat_length, channel)
        ECG_Data = np.concatenate((ECG_Data, ecg_data), axis=0)

        cur_label = torch.from_numpy(np.int64(cur_label))
        # RR_feature = np.concatenate((RR_feature, rr_feature), axis=0)
        cur_label = torch.nn.functional.one_hot(cur_label , 5)
        Label = np.concatenate((Label, cur_label.numpy()), axis=0)

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

    feature_num = 1
    channel = 1
    # left_beat = 70
    # right_beat = 99
    beat_length = 251
    '''not AAMI standard, but most papers like this'''
    # 标签重编码
    label_t5 = {'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
                'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
                'V': 'V', 'E': 'V',
                'F': 'F',
                '/': 'Q', 'f': 'Q', 'Q': 'Q'
                }

    num = 0
    for cur_record_id in data_list:
        print(cur_record_id)
        file_name = os.path.join(data_dir, str(cur_record_id)) + '.hea'
        ECG_Data = np.array([]).reshape((0, feature_num, beat_length, channel))
        ECG_Data_train = np.array([]).reshape((0, feature_num, beat_length, channel))
        Label = np.array([]).reshape((0, 5))
        Label_train = np.array([]).reshape((0, 5))

        one_beat, cur_label,train_beat,train_label = heartbeat_uda(file_name, data_dir)

        ecg_data = np.concatenate((one_beat), axis=0)
        ecg_data_train = np.concatenate((train_beat), axis=0)
        ecg_data = ecg_data.reshape(-1, feature_num, beat_length, channel)
        ecg_data_train = ecg_data_train.reshape(-1, feature_num, beat_length, channel)
        ECG_Data = np.concatenate((ECG_Data, ecg_data), axis=0)
        ECG_Data_train = np.concatenate((ECG_Data_train, ecg_data_train), axis=0)

        cur_label = torch.from_numpy(np.int64(cur_label))
        train_label = torch.from_numpy(np.int64(train_label))

        # RR_feature = np.concatenate((RR_feature, rr_feature), axis=0)
        cur_label = torch.nn.functional.one_hot(cur_label, 5)
        train_label= torch.nn.functional.one_hot(train_label, 5)
        Label = np.concatenate((Label, cur_label.numpy()), axis=0)
        Label_train = np.concatenate((Label_train, train_label.numpy()), axis=0)

        print(ECG_Data.shape)
        print(Label.shape)
        print(ECG_Data_train.shape)
        print(Label_train.shape)
        # np.savez(save_dir+str(cur_record_id), ECG_Data=ECG_Data, RR_feature=RR_feature, Label=Label)
        np.savez(DS2_test_save_dir+ str(cur_record_id), ECG_Data=ECG_Data, Label=Label)
        np.savez(DS2_train_save_dir+ str(cur_record_id), ECG_Data=ECG_Data_train, Label=Label_train)
        num = num + 1
        # # if num == 1:
        # #     break

        print('=====Processing第' + str(num) + '个record=====\n')

#专门准备DS2测试数据和训练数据，前5min数据为SVDB\train   之后的数据为SVDB\test
def svdb_data_prepare_single_128samples_beat_for_UDA(data_dir, data_list, save_dir):
    folder_name = save_dir.split('/')[2] + '/'
    Heartbeats_img = 'exp1/HeartBeatsImg/' + folder_name
    if os.path.exists(Heartbeats_img):
        shutil.rmtree(Heartbeats_img)
    if not os.path.exists(Heartbeats_img):
        os.makedirs(Heartbeats_img)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    feature_num = 1
    channel = 1
    # left_beat = 70
    # right_beat = 99
    beat_length = 251
    '''not AAMI standard, but most papers like this'''
    # 标签重编码
    label_t5 = {'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
                'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
                'V': 'V', 'E': 'V',
                'F': 'F',
                '/': 'Q', 'f': 'Q', 'Q': 'Q'
                }

    num = 0
    for cur_record_id in data_list:
        print(cur_record_id)
        file_name = os.path.join(data_dir, str(cur_record_id)) + '.hea'
        ECG_Data = np.array([]).reshape((0, feature_num, beat_length, channel))
        ECG_Data_train = np.array([]).reshape((0, feature_num, beat_length, channel))
        Label = np.array([]).reshape((0, 5))
        Label_train = np.array([]).reshape((0, 5))

        one_beat, cur_label,train_beat,train_label = heartbeat_uda(file_name, data_dir)

        ecg_data = np.concatenate((one_beat), axis=0)
        ecg_data_train = np.concatenate((train_beat), axis=0)
        ecg_data = ecg_data.reshape(-1, feature_num, beat_length, channel)
        ecg_data_train = ecg_data_train.reshape(-1, feature_num, beat_length, channel)
        ECG_Data = np.concatenate((ECG_Data, ecg_data), axis=0)
        ECG_Data_train = np.concatenate((ECG_Data_train, ecg_data_train), axis=0)

        cur_label = torch.from_numpy(np.int64(cur_label))
        train_label = torch.from_numpy(np.int64(train_label))

        # RR_feature = np.concatenate((RR_feature, rr_feature), axis=0)
        cur_label = torch.nn.functional.one_hot(cur_label, 5)
        train_label= torch.nn.functional.one_hot(train_label, 5)
        Label = np.concatenate((Label, cur_label.numpy()), axis=0)
        Label_train = np.concatenate((Label_train, train_label.numpy()), axis=0)

        print(ECG_Data.shape)
        print(Label.shape)
        print(ECG_Data_train.shape)
        print(Label_train.shape)
        # np.savez(save_dir+str(cur_record_id), ECG_Data=ECG_Data, RR_feature=RR_feature, Label=Label)
        np.savez(save_dir+'/test/'+ str(cur_record_id), ECG_Data=ECG_Data, Label=Label)
        np.savez(save_dir +'/train/'+ str(cur_record_id), ECG_Data=ECG_Data_train, Label=Label_train)
        num = num + 1
        # # if num == 1:
        # #     break

        print('=====Processing第' + str(num) + '个record=====\n')

def main():
    # '''Prepare exp data: DS1->DS2'''
    ##data_dir存放的文件包括xxx.atr, xxx.data, xxx.hea...
    data_dir = "./MIT-BIH"
    train_record_list = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203,
                         205, 207, 208, 209, 215, 220, 223, 230]
    test_record_list = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214,
                        219, 221, 222, 228, 231, 232, 233, 234]
    # 将读取出的每个reocrd的data和label保存为npz文件
    train_save_dir = 'data_new/mitdb_uda_DS1/'
    test_save_dir = 'data_new/mitdb_uda_DS2/'

    mitdb_data_prepare_single_128samples_beat(data_dir, train_record_list, train_save_dir)
    mitdb_data_prepare_single_128samples_beat_for_UDA(data_dir, test_record_list, test_save_dir)



    '''Prepare exp data: MITDB->SVDB'''
    data_dir = "./MIT-BIH"
    train_save_dir = 'data_new/mitdb_DS1_DS2_whole_data/'
    records_list = train_record_list + test_record_list
    mitdb_data_prepare_single_128samples_beat(data_dir, records_list, train_save_dir)

    data_dir = "./SVDB"
    test_save_dir = 'data_new/svdb_uda_128hz/'
    data_list = os.listdir(data_dir)
    data_list = [k for k in data_list if k.startswith('8')]##svdb
    data_list = np.unique([k[:3] for k in data_list ])
    print('The number of svdb records:', len(data_list))
    print(data_list)
    mitdb_data_prepare_single_128samples_beat_for_UDA(data_dir, data_list, test_save_dir)


if __name__ == "__main__":
    main()
