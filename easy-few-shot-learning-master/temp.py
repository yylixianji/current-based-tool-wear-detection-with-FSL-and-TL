import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
cm=np.array([[14,1],
             [1,14]])

plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, cmap='Greens')
plt.ylim(0, 2)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()
