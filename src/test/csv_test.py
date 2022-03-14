import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import mpl



x_tick=['a','b','c']
y_tick=['x','y','z']
X=[[1,2,3],[4,5,6],[7,8,9]]
data={}
for i in range(3):
    data[x_tick[i]] = X[i]
pd_data=pd.DataFrame(data,index=y_tick,columns=x_tick)
print(pd_data)

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
font = {'family': 'sans-serif',
        'color': 'k',
        'weight': 'normal',
        'size': 20, }

f, ax = plt.subplots(figsize=(8, 8))
cmap = sns.cm.rocket_r  # colorbar颜色反转
ax = sns.heatmap(pd_data, annot=True, ax=ax, fmt='.1f', cmap=cmap)  # 画heatmap，具体参数可以查文档

plt.xlabel('x_label', fontsize=20, color='k')  # x轴label的文本和字体大小
plt.ylabel('y_label', fontsize=20, color='k')  # y轴label的文本和字体大小
plt.xticks(fontsize=20)  # x轴刻度的字体大小（文本包含在pd_data中了）
plt.yticks(fontsize=20)  # y轴刻度的字体大小（文本包含在pd_data中了）
plt.title('title', fontsize=20)  # 图片标题文本和字体大小
# 设置colorbar的刻度字体大小
cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=20)
# 设置colorbar的label文本和字体大小
cbar = ax.collections[0].colorbar
cbar.set_label(r'$NMI$', fontdict=font)

plt.show()