import pandas as pd
import numpy as np
df = pd.read_csv('arbeit data/experiment/N_200to400_y0.1_s200_01-50.csv',
                 engine='python',
                 dtype=np.float64,
                 skipfooter=1,
                 delim_whitespace=True,
                 names=["time", "value"],
                 index_col="time",
                 header=20)  # Path of the .csv file
img = df.plot(title="N_200to400_y0.1_s200_01-50", xlabel="time", ylabel="value")
fig = img.get_figure()
fig.savefig(r'a.png')
