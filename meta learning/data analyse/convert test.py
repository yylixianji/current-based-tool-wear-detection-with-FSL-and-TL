import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# data
data = pd.read_csv('d12 1edge.csv',
                   # engine='python',
                   # dtype=np.float64,
                   # skipfooter=1,
                   skiprows=1,
                   # delim_whitespace=True,
                   # names=["time", "value"],
                   header= None,
                   index_col=0,
                   )  # Path of the .csv file

data.plot()
plt.show()
print(data)  # to check the shape
print(data.head(5))  # Use this to print the first 5 lines of the data, to understand it better

