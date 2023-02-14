import pandas as pd
import numpy as np

# 打开文件
data = pd.read_csv('new data/new data current6/d20 3edges.csv',
                   skiprows=33100,
                   # header=0,
                   # index_col=0
                   )  # Path of the .csv file
# 每个excel保存3万行，那么530000+数据需要18个.csv文档保存
for i in range(0, 1):
    save_data = data.iloc[i * 22200 + 1: (i + 1) * 22200 + 1]
    file_name = 'new data/splited/d20 3edges' + str(i+2) + '.csv'  # 保存文件路径以及文件名称
    save_data.to_csv(file_name, index=False)  # 保存格式为.csv，如果是xlsx则修改为save_data.to_excel
