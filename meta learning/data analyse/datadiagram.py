import os
import pandas as pd


def get_files(path='E:\\mit\\masterarbeit\\meta learning\\data analyse\\24', rule=".csv"):
    for fpathe, dirs, fs in os.walk(path):
        for f in fs:
            filename = os.path.join(fpathe, f)
            if filename.endswith(rule):
                with open(filename, encoding='UTF-8-sig') as csvfile:
                    df = pd.read_csv(csvfile,

                                     index_col="time")
                    img = df.plot(xlabel="time", ylabel="value", legend=False)
                    fig = img.get_figure()
                    fig.savefig(filename.replace(".csv", ".jpg"))


get_files()
