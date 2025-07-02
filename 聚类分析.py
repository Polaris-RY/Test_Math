import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import matplotlib

# 设置 Matplotlib 后端
matplotlib.use('TkAgg')

# 设置matplotlib支持中文显示
rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取数据
train_data = pd.read_csv('train.csv', encoding='GBK')

# 使用K-means进行聚类分析
# 将洪水概率列提取出来进行聚类
X = train_data[['洪水概率']]

# 使用K-means聚类
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
train_data['风险类别'] = kmeans.labels_

# 可视化聚类结果
plt.figure(figsize=(10, 6))
sns.scatterplot(x=train_data.index, y='洪水概率', hue='风险类别', data=train_data, palette='viridis')
plt.title('洪水概率聚类结果')
plt.show()

# 分析不同风险类别的指标特征
high_risk = train_data[train_data['风险类别'] == 0]
medium_risk = train_data[train_data['风险类别'] == 1]
low_risk = train_data[train_data['风险类别'] == 2]

print("High risk group:\n", high_risk.describe())
print("Medium risk group:\n", medium_risk.describe())
print("Low risk group:\n", low_risk.describe())