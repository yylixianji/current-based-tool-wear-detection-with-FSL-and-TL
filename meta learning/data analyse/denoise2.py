import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt


# sgn函数
def sgn(num):
    if num > 0.0:
        return 1.0
    elif num == 0.0:
        return 0.0
    else:
        return -1.0


def wavelet_noising(new_df):
    data = new_df
    data = data.values.T.tolist()  # 将np.ndarray()转为列表
    w = pywt.Wavelet('dB1')  # 选择dB10小波基
    ca10, cd10, cd9, cd8, cd7, cd6, cd5, cd4, cd3, cd2, cd1 = pywt.wavedec(data, w, level=10)  # 3层小波分解
    ca10 = ca10.squeeze(axis=0)
    cd10 = cd10.squeeze(axis=0)
    cd9 = cd9.squeeze(axis=0)
    cd8 = cd8.squeeze(axis=0)
    cd7 = cd7.squeeze(axis=0)
    cd6 = cd6.squeeze(axis=0)
    cd5 = cd5.squeeze(axis=0)
    cd4 = cd4.squeeze(axis=0)  # ndarray数组减维：(1，a)->(a,)
    cd3 = cd3.squeeze(axis=0)
    cd2 = cd2.squeeze(axis=0)
    cd1 = cd1.squeeze(axis=0)
    length1 = len(cd1)
    length0 = len(data[0])

    abs_cd1 = np.abs(np.array(cd1))
    median_cd1 = np.median(abs_cd1)

    sigma = (1.0 / 0.6745) * median_cd1
    lamda = sigma * math.sqrt(2.0 * math.log(float(length0), math.e))
    usecoeffs = []
    usecoeffs.append(ca10)

    # 软阈值方法
    for k in range(length1):
        if abs(cd1[k]) >= lamda / np.log2(2):
            cd1[k] = sgn(cd1[k]) * (abs(cd1[k]) - lamda / np.log2(2))
        else:
            cd1[k] = 0.0

    length2 = len(cd2)
    for k in range(length2):
        if abs(cd2[k]) >= lamda / np.log2(3):
            cd2[k] = sgn(cd2[k]) * (abs(cd2[k]) - lamda / np.log2(3))
        else:
            cd2[k] = 0.0

    length3 = len(cd3)
    for k in range(length3):
        if abs(cd3[k]) >= lamda / np.log2(4):
            cd3[k] = sgn(cd3[k]) * (abs(cd3[k]) - lamda / np.log2(4))
        else:
            cd3[k] = 0.0
    length4 = len(cd4)
    for k in range(length4):
        if abs(cd4[k]) >= lamda / np.log2(5):
            cd4[k] = sgn(cd4[k]) * (abs(cd4[k]) - lamda / np.log2(5))
        else:
            cd4[k] = 0.0
    length5 = len(cd5)
    for k in range(length5):
        if abs(cd5[k]) >= lamda / np.log2(6):
            cd5[k] = sgn(cd5[k]) * (abs(cd5[k]) - lamda / np.log2(6))
        else:
            cd5[k] = 0.0
    length6 = len(cd6)
    for k in range(length6):
        if abs(cd6[k]) >= lamda / np.log2(7):
            cd6[k] = sgn(cd6[k]) * (abs(cd6[k]) - lamda / np.log2(7))
        else:
            cd6[k] = 0.0
    length7 = len(cd7)
    for k in range(length7):
        if abs(cd7[k]) >= lamda / np.log2(8):
            cd7[k] = sgn(cd7[k]) * (abs(cd7[k]) - lamda / np.log2(8))
        else:
            cd7[k] = 0.0
    length8 = len(cd8)
    for k in range(length8):
        if abs(cd8[k]) >= lamda / np.log2(9):
            cd8[k] = sgn(cd8[k]) * (abs(cd8[k]) - lamda / np.log2(9))
        else:
            cd8[k] = 0.0
    length9 = len(cd9)
    for k in range(length9):
        if abs(cd9[k]) >= lamda / np.log2(10):
            cd9[k] = sgn(cd9[k]) * (abs(cd9[k]) - lamda / np.log2(10))
        else:
            cd9[k] = 0.0
    length10 = len(cd10)
    for k in range(length10):
        if abs(cd10[k]) >= lamda / np.log2(11):
            cd10[k] = sgn(cd10[k]) * (abs(cd10[k]) - lamda / np.log2(11))
        else:
            cd10[k] = 0.0

    usecoeffs.append(cd10)
    usecoeffs.append(cd9)
    usecoeffs.append(cd8)
    usecoeffs.append(cd7)
    usecoeffs.append(cd6)
    usecoeffs.append(cd5)
    usecoeffs.append(cd4)
    usecoeffs.append(cd3)
    usecoeffs.append(cd2)
    usecoeffs.append(cd1)
    recoeffs = pywt.waverec(usecoeffs, w)  # 信号重构
    return recoeffs


# 主函数
path = 'small csv/new Figure/d12 1edge1.csv'  # 数据路径

# 提取数据
data = pd.read_csv(path, usecols=[1])
data = data.iloc[:]
plt.figure()
plt.plot(data)

data_denoising = wavelet_noising(data)  # 调用小波阈值方法去噪
print(data_denoising)
plt.figure()
plt.plot(data_denoising)  # 显示去噪结果
plt.show()
