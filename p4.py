import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib
import joblib

# 加载 scaler_y 和 lambda_
scaler_y = joblib.load('scaler_y.pkl')
lambda_ = np.load('lambda_.npy')

# 设置Matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# 定义FloodNet模型
class FloodNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(FloodNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layer(x)

# 数据预处理函数
def min_max_scaler(data):
    return (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-8)

# 加载训练好的模型一
input_dim = 20
model_1 = FloodNet(input_dim)

try:
    # 加载模型参数
    state_dict = torch.load('model_1.pth', weights_only=True)
    # 筛选匹配的参数
    model_dict = model_1.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    # 更新模型参数
    model_dict.update(pretrained_dict)
    model_1.load_state_dict(model_dict)
    print("部分参数加载成功")
except Exception as e:
    print(f"无法加载模型参数，请检查模型文件。错误信息：{e}")

model_1.eval()

# 读取测试数据
try:
    # 确保文件路径正确，如果文件不在当前工作目录下，需要提供完整路径
    test_data = pd.read_csv("test.csv", encoding="gbk")
    print(f"测试数据加载成功，共{len(test_data)}条记录，{len(test_data.columns)}个特征")
except Exception as e:
    print(f"无法加载测试数据，请检查文件路径。错误信息：{e}")

# 数据预处理
# 检查并处理缺失值
if test_data.isnull().any().any():
    print("警告：测试数据包含缺失值，进行简单填充...")
    test_data = test_data.fillna(test_data.mean())

# 特征提取
X_test = test_data.values
# 归一化
X_test_scaled = min_max_scaler(X_test)
# 转为Tensor
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

# 进行预测
with torch.no_grad():
    y_pred = model_1(X_test_tensor)
    # 假设 y_pred_1 是模型1的预测结果
    y_pred_1_unscaled = scaler_y.inverse_transform(y_pred.numpy())
    y_pred_np = y_pred.numpy().flatten()
    # 对逆变换后的结果进行Box-Cox逆变换
    from scipy.special import inv_boxcox
    y_pred_1_final = inv_boxcox(y_pred_1_unscaled, lambda_) - 1

# 保存预测结果到submit.csv
submit_data = pd.DataFrame(y_pred_np, columns=['洪水概率'])
submit_data.to_csv('submit.csv', index=False)

# 绘制直方图
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.hist(y_pred_np, bins=50, edgecolor='k')
plt.title('洪水概率直方图')
plt.xlabel('洪水概率')
plt.ylabel('频数')

# 绘制折线图
plt.subplot(1, 3, 2)
plt.plot(np.sort(y_pred_np))
plt.title('洪水概率折线图')
plt.xlabel('事件编号')
plt.ylabel('洪水概率')

# QQ图
plt.subplot(1, 3, 3)
stats.probplot(y_pred_np, dist="norm", plot=plt)
plt.title('QQ图')
plt.xlabel('理论分位数')
plt.ylabel('样本分位数')

# 保存图像到文件
plt.tight_layout()
plt.savefig('flood_probability_plots.png')  # 保存图像到文件
print("图像已保存到flood_probability_plots.png")

# Shapiro-Wilk检验
stat, p = stats.shapiro(y_pred_np)
print(f"Shapiro-Wilk检验统计量: {stat:.6f}, p值: {p:.6f}")
if p > 0.05:
    print("数据可能服从正态分布。")
else:
    print("数据不服从正态分布。")

# Kolmogorov-Smirnov检验
stat, p = stats.kstest(y_pred_np, 'norm')
print(f"Kolmogorov-Smirnov检验统计量: {stat:.6f}, p值: {p:.6f}")
if p > 0.05:
    print("数据可能服从正态分布。")
else:
    print("数据不服从正态分布。")

# Anderson-Darling检验
result = stats.anderson(y_pred_np)
print(f"Anderson-Darling检验统计量: {result.statistic:.6f}")
for i in range(len(result.critical_values)):
    sl, cv = result.significance_level[i], result.critical_values[i]
    if result.statistic < cv:
        print(f"在显著性水平{sl}下，数据可能服从正态分布。")
    else:
        print(f"在显著性水平{sl}下，数据不服从正态分布。")