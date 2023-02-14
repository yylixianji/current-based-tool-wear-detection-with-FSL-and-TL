import json
import os
import matplotlib.pyplot as plt
import pandas as pd

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 1000000)
file = open("d16 3edges.json", "r")
data = file.read()
data = json.loads(data)
# df1 = json_normalize(data, ['Header', 'SignalListHFData'])
# df2 = json_normalize(data, ['Payload'], errors='ignore')
# print(df2)

df1 = pd.DataFrame.from_dict(data['Header']['SignalListHFData'])
df1 = df1.T
print(df1)
df2 = pd.DataFrame.from_dict(data['Payload'])
print(df2)
df3 = df2.drop([
    'HFCallEvent',
    'HFBlockEvent'], axis=1)
df3 = df3.dropna(how="all")
df4 = pd.DataFrame(df3.explode(['HFData'])['HFData'].to_list(), columns=df1.iloc[3])
df4 = df4[['CURRENT|6']]
df4.plot()
plt.xlim((-2500, 60000))
plt.ylim((-4, 1))
plt.xticks([0,10000,20000,30000,40000,50000,60000],
          [r'$0$', r'$20$', r'$40$', r'$60$', r'$80$', r'$100$', r'$120$'])
plt.legend(labels=['S-axis motor current'],loc='best')
plt.xlabel('Time/s')
plt.ylabel('Current/A')
plt.show()
df4.to_csv('new data current2/d12 normal2.csv')
# 'HFCallEvent'
