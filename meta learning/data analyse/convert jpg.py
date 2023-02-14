import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# path='E:\\mit\\masterarbeit\\meta learning\\data analyse\\new data\\new data json
def get_files(path='E:\\mit\\masterarbeit\\meta learning\\data analyse\\small csv test', rule=".csv"):
    for fpathe, dirs, fs in os.walk(path):
        for f in fs:
            filename = os.path.join(fpathe, f)
            if filename.endswith(rule):
                with open(filename, encoding='UTF-8-sig') as csvfile:
                    list = pd.read_csv(csvfile,
                                       # engine='python',
                                       dtype=np.float64,
                                       # skipfooter=1,
                                       # delim_whitespace=True,
                                       skiprows=1,
                                       header=None,
                                       index_col=0,
                                       # names=["time", "value"],
                                       # header=20
                                       )
                    print(list)
                    img = list.plot()
                    # img.get_legend().remove()
                    # img.axis('off')
                    plt.xlim((0, 180000))
                    plt.ylim((-5, 1))
                    plt.xticks([0, 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000],
                               [r'$0$', r'$40$', r'$80$', r'$120$', r'$160$', r'$200$', r'$240$', r'$280$', r'$320$', r'$360$'])
                    plt.legend(labels=['S-axis motor current',
                                       # 'Y-axis motor current',
                                       # 'S-axis motor current'
                                       ], loc='best')
                    plt.xlabel('Time/s')
                    plt.ylabel('Current/A')
                    fig = img.get_figure()
                    fig.savefig(filename.replace(".csv", ".jpg"))


get_files()
