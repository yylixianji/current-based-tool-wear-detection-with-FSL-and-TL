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
                                       skiprows=76)  # Path of the .csv file
                    for i in range(0, 78):
                        save_data = data.iloc[i * 338 + 1: (i + 1) * 338 + 1]
                        file_name = filename.split('.')[0] + '-' + str(i) + '.csv'  # 保存文件路径以及文件名称
                        save_data.to_csv(file_name, index=False)  # 保存格式为.csv，如果是xlsx则修改为save_data.to_excel


get_files()
