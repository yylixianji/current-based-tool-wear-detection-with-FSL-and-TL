from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
y_true = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
y_pred = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 8))
# cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]     # 归一化
# cm_norm = np.around(cm_norm, decimals=2)
sns.set(font_scale=1.8)
sns.heatmap(cm, annot=True, cmap='Blues')

plt.ylim(0, 2)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()
