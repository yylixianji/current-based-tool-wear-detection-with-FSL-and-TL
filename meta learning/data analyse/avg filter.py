"""
均值滤波降噪：
    函数ava_filter用于单次计算给定窗口长度的均值滤波
    函数denoise用于指定次数调用ava_filter函数，进行降噪处理
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from statsmodels.nonparametric.smoothers_lowess import lowess
# plt.rcParams.update({'xtick.bottom' : False, 'axes.titlepad':5})

# Import
# df_orig = pd.read_csv('small csv/new Figure/d12 1edge1.csv', parse_dates=['date'], index_col='date')
#
# # 1. Moving Average
# df_ma = df_orig.value.rolling(3, center=True, closed='both').mean()
#
# # 2. Loess Smoothing (5% and 15%)
# df_loess_5 = pd.DataFrame(lowess(df_orig.value, np.arange(len(df_orig.value)), frac=0.01)[:, 1], index=df_orig.index, columns=['value'])
# # df_loess_15 = pd.DataFrame(lowess(df_orig.value, np.arange(len(df_orig.value)), frac=0.15)[:, 1], index=df_orig.index, columns=['value'])
#
# # Plot
# # fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True, dpi=640)
# # df_orig['value'].plot(ax=axes[0], color='k', title='Original Series')
# df_loess_5['value'].plot(title='Loess Smoothed 1%')
# # df_loess_15['value'].plot(ax=axes[2], title='Loess Smoothed 15%')
# # df_ma.plot(ax=axes[3], title='Moving Average (3)')
# # fig.suptitle('How to Smoothen a Time Series', y=0.95, fontsize=14)
# plt.show()

def get_files(path='E:\\mit\\masterarbeit\\meta learning\\data analyse\\small csv test', rule=".csv"):
    for fpathe, dirs, fs in os.walk(path):
        for f in fs:
            filename = os.path.join(fpathe, f)
            if filename.endswith(rule):
                with open(filename, encoding='UTF-8-sig') as csvfile:
                    list = pd.read_csv(csvfile, parse_dates=['date'], index_col='date'
                                       )
                    plt.rcParams.update({'xtick.bottom': False, 'axes.titlepad': 5})
                    df_loess_5 = pd.DataFrame(lowess(list.value, np.arange(len(list.value)), frac=0.01)[:, 1],
                                              index=list.index, columns=['value'])
                    df_loess_5['value'].plot()
                    plt.axis('off')
                    plt.savefig(filename.replace(".csv", ".jpg"))
                    plt.close()


get_files()
