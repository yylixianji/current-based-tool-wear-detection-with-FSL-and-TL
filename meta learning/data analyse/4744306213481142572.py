
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import pywt 

####################一些参数和函数############
def sgn(num):
    if(num > 0.0):
        return 1.0
    elif(num == 0.0):
        return 0.0
    else:
        return -1.0

###软硬阈值折衷法 a 参数
#read data
data = pd.read_csv('small csv/new Figure/d12 1edge1.csv', usecols=[1])
#y_value为原信号
a=0.5
# x1 = np.array(data)
y_values = np.array(data)

#plt.subplot(211)
#plt.scatter(x1, y_values, s=10)
#小波基的选取
w = pywt.Wavelet('db1')#选用db5小波
#ca3, cd3, cd2, cd1 = pywt.wavedec(y_values, w)
maxlev = pywt.dwt_max_level(len(data), w)#最大分解级别，返回max_level。db.dec_lenx为小波的长度
coeffs = pywt.wavedec(y_values, w, mode='constant',level=maxlev)#分解波
#recoeffs = pywt.waverec(coeffs, w)#多级重建

 #求通用阈值
thcoeffs =[]
for i in range(1, len(coeffs)):
    tmp = coeffs[i].copy()
    Sum = 0.0
    for j in coeffs[i]:
        Sum = Sum + abs(j)
    print(Sum)
    N = len(coeffs[i])
    Sum = (1.0 / float(N)) * Sum
    sigma = (1/0.6745)*Sum   
    lamda = sigma * math.sqrt(2.0 * math.log(float(N), math.e)) #采用通用的阈值
    for k in range(len(tmp)):
        if(abs(tmp[k]) >= lamda):
            tmp[k] = sgn(tmp[k]) * (abs(tmp[k]) - a*lamda)
        else:
            tmp[k] = 0.0
    thcoeffs.append(tmp)

#print(thcoeffs)
usecoeffs = []
usecoeffs.append(thcoeffs[0])
usecoeffs.extend(thcoeffs)
#recoeffs为去噪后信号
#重组信号
recoeffs = pywt.waverec(usecoeffs, w)

plt.subplot(212) 
plt.plot(recoeffs)       
plt.show()
 





