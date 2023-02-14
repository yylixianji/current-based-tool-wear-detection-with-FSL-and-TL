import os

import pandas as pd


def get_files(path='E:\\mit\\masterarbeit\\meta learning\\data analyse\\24', rule=".csv"):
    for fpathe, dirs, fs in os.walk(path):
        for f in fs:
            filename = os.path.join(fpathe, f)
            if filename.endswith(rule):
                with open(filename, encoding='UTF-8-sig') as csvfile:
                    data = pd.read_csv(csvfile,
                                       engine='python',
                                       skipfooter=2,
                                       delim_whitespace=True,
                                       names=["time", "value"],
                                       # index_col="time",
                                       skiprows=24)  # Path of the .csv file
                    # 每个excel保存3万行，那么530000+数据需要18个.csv文档保存
                    data.to_csv(filename, index=False)  # 保存格式为.csv，如果是xlsx则修改为save_data.to_excel


get_files()
