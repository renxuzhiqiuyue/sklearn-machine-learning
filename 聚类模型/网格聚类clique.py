import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn.datasets as ds

## 设置属性防止中文乱码及拦截异常信息
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

x = np.arange(0, 2, step=0.1).tolist() + np.arange(4, 6, step = 0.2).tolist() + [0.3]
y = np.random.uniform(0, 1.98, size = 20).tolist() + np.random.uniform(4, 5.98, size = 10).tolist() + [5.3]

plt.plot(x, y, 'bo')
plt.grid(linestyle='-.')
plt.show()

# 选择聚类方法：clique 类
from pyclustering.cluster.clique import clique
# clique 可视化
from pyclustering.cluster.clique import clique_visualizer

data = np.array([x, y]).T

# 创建 CLIQUE 算法进行处理
# 定义每个维度中网格单元的数量
intervals = 6
# 密度阈值
threshold = 1
clique_instance = clique(data, intervals, threshold)

# 开始聚类过程并获得结果
clique_instance.process()
clique_cluster = clique_instance.get_clusters()  # allocated clusters

# 被认为是异常值的点（噪点）
noise = clique_instance.get_noise()
# CLIQUE形成的网格单元
cells = clique_instance.get_cells() 

print("Amount of clusters:", len(clique_cluster))
print(clique_cluster)
print(noise)

clique_visualizer.show_grid(cells, data)
clique_visualizer.show_clusters(data, clique_cluster, noise) # show clustering results