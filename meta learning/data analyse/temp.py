import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from statsmodels.nonparametric.smoothers_lowess import lowess
def get_files(path='E:\\mit\\masterarbeit\\meta learning\\data analyse\\small csv\\new2', rule=".csv"):
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