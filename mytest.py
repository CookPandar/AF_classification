# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 17:59:42 2022

@author: 张树文
"""
import re
import random as rd
import numpy as np
import pandas as pd
import warnings

warnings.simplefilter("ignore")

def preprocess(fd):
    #去掉头5行
    fd = fd.drop(index=[0, 1, 2, 3, 4])
    fd = fd.reset_index(drop=True)
    # 去掉有图片的
    well = fd.iloc[:, 8].str.contains('http', na=False)
    well = np.where(well == True)[0]
    fd = fd.drop(index=well)
    fd = fd.reset_index(drop=True)
    fd = fd.dropna(how='all', subset=['Unnamed: 8'])
    fd = fd.reset_index(drop=True)
    # 处理形如  '1','2','3' 的行
    # 虽然只有一个

    # 1. 匹配 ‘   。这里用’\’‘转义 ‘
    new_col = fd.iloc[:, 8].str.contains('\'', na=False)
    new_col = np.where(new_col == True)[0]

    # 2. 将有’‘的行的元素依次依据’拆分，提取其中的数字变成一个列表
    for i in new_col:
        c = fd.iloc[i, 8]
        d = fd.iloc[i, 9]
        my_c = c.split('\'')
        my_d = d.split('\'')
        new_list = []
        while ('' in my_c):
            my_c.remove('')
        while ('\'' in my_c):
            my_c.remove('\'')
        while (', ' in my_c):
            my_c.remove(', ')

        while ('' in my_d ):
            my_d .remove('')
        while ('\'' in my_d ):
            my_d .remove('\'')
        while (', ' in my_d ):
            my_d .remove(', ')
        # 将列表转化成字符串
        str = ','.join(my_c)
        fd.iloc[i, 8] = str
        str = ','.join(my_d)
        fd.iloc[i, 9] = str
    fd = fd.reset_index(drop=True)
    list = []
    #少于14的先补全0
    for i in range(0, fd.shape[0]):
        # if len(fd.iloc[i, 8]) != 14:
        #     list.append(i)
        # if len(fd.iloc[i, 8]) < 14:
        #     c = fd.iloc[i, 8]
        #     d = fd.iloc[i, 9]
        #
        #     my_c = c.split('\'')
        #     my_d = d.split('\'')
        #     new_list = []
        #     while ('， ' in my_c):
        #         my_c.remove('， ')
        #     while ('，' in my_c):
        #         my_c.remove('，')
        #     while (' ，' in my_c):
        #         my_c.remove(' ，')
        #     while ('' in my_c):
        #         my_c.remove('')
        #     while ('\'' in my_c):
        #         my_c.remove('\'')
        #     while (', ' in my_c):
        #         my_c.remove(', ')
        #
        #     while ('， ' in my_d):
        #         my_d.remove('， ')
        #     while ('，' in my_d):
        #         my_d.remove('，')
        #     while (' ，' in my_d):
        #         my_d.remove(' ，')
        #     while ('' in my_d):
        #         my_d.remove('')
        #     while ('\'' in my_d):
        #         my_d.remove('\'')
        #     while (', ' in my_d):
        #         my_d.remove(', ')
        #     index=0
        #     for  m in  my_c:
        #         if len(m)!=2:
        #             m='0'+m
        #         my_c[index] = m
        #         index=index+1
        #     index = 0
        #     for m in  my_d:
        #         if len(m)==1:
        #             m='0'+m
        #         my_d[index] = m
        #         index = index + 1
        #     # 将列表转化成字符串
        #     str = ','.join(my_c)
        #     fd.iloc[i, 8] = str
        #     str=[]
        #     str = ','.join(my_d)
        #     fd.iloc[i, 9] = str
        result = re.findall("^\d{2},\d{2},\d{2},\d{2},\d{2}$", fd.iloc[i, 8])
        if result ==[]:
            list.append(i)

        # elif len(fd.iloc[i, 8])>14:
        #     list.append(i)

    fd = fd.drop(index=list)
    fd = fd.reset_index(drop=True)

    #转成两位数
    #第9 10行转为数字

    print(fd)
    return fd



def readnum():
    name=fd.iloc[5:,0].values  #第六行到最后 第一列
    # name=fd.iloc[5:,8].dropna(axis=0,how='any')
    # name=fd.drop(fd[:,9].str.contains('http',na=False))
    f5 = fd.iloc[5:, 8].values  # 第六行到最后  第九列
    b2 = fd.iloc[5:, 9].values  # 第六行到最后   第十列
    return name, f5, b2


if __name__ == '__main__':
    fd = pd.read_excel('C:/Users/张树文/Documents/Tencent Files/2224714732/FileRecv/体彩大乐透-问卷统计详情.xlsx')

    fd = preprocess(fd)
    fd.to_excel("处理后.xlsx")





