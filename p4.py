import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib
import joblib
import sys
import seaborn as sns

# 加载 scaler_y 和 lambda_
scaler_y = joblib.load('scaler_y.pkl')
lambda_ = np.load('lambda_.npy')

# 设置Matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# 重新定义模型结构，与训练代码一致
class FloodNetV3(nn.Module):
    def __init__(self, input_dim):
        super(FloodNetV3, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)


# 读取测试数据
try:
    test_data = pd.read_csv("test.csv", encoding="gbk")
    print(f"测试数据加载成功，共{len(test_data)}条记录，{len(test_data.columns)}个特征")
except Exception as e:
    print(f"无法加载测试数据: {e}")
    sys.exit(1)

# 数据预处理
if test_data.isnull().any().any():
    print("警告：测试数据包含缺失值，进行简单填充...")
    test_data = test_data.fillna(test_data.mean())

# 关键修复：确保特征数量与训练时一致
X_test = test_data.iloc[:, :21].values  # 只取前20个特征

# 加载训练时使用的scaler_X
try:
    scaler_X = joblib.load('scaler_X.pkl')
    print("成功加载训练时使用的标准化器")
except:
    print("错误：未找到scaler_X.pkl，无法继续")
    sys.exit(1)

# 使用相同的标准化方法
X_test_scaled = scaler_X.transform(X_test)

# 转为Tensor
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

# 动态设置输入维度
input_dim = X_test.shape[1]
print(f"模型输入维度设置为: {input_dim}")

# 加载训练好的模型
model_1 = FloodNetV3(input_dim)

try:
    # 加载模型参数
    state_dict = torch.load('flood_prediction_model.pth')
    model_1.load_state_dict(state_dict)
    print("✅ 模型参数完整加载成功")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    sys.exit(1)

model_1.eval()
print("模型设置为评估模式")

# 进行预测
with torch.no_grad():
    print("开始预测...")
    y_pred = model_1(X_test_tensor)
    y_pred_np = y_pred.numpy().flatten()

    # 逆标准化
    y_pred_1_unscaled = scaler_y.inverse_transform(y_pred_np.reshape(-1, 1))

    # Box-Cox逆变换
    from scipy.special import inv_boxcox

    try:
        y_pred_1_final = inv_boxcox(y_pred_1_unscaled, lambda_) - 1
    except Exception as e:
        print(f"Box-Cox逆变换错误: {e}")
        y_pred_1_final = y_pred_1_unscaled

    print("预测完成")

# 保存预测结果
submit_data = pd.DataFrame(y_pred_1_final, columns=['洪水概率'])
submit_data.to_csv('submit.csv', index=False)
print("预测结果已保存到submit.csv")

# 可视化改进：添加更多图表类型
plt.figure(figsize=(15, 10))
# 1. Flood Probability Histogram
plt.subplot(2, 2, 1)
plt.hist(y_pred_1_final, bins=50, edgecolor='k', alpha=0.7)
plt.title('Flood Probability Distribution Histogram')
plt.xlabel('Flood Probability')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.7)

# 2. Cumulative Distribution Plot
plt.subplot(2, 2, 2)
sorted_values = np.sort(y_pred_1_final.flatten())
cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
plt.plot(sorted_values, cdf, linewidth=2)
plt.title('Flood Probability Cumulative Distribution')
plt.xlabel('Flood Probability')
plt.ylabel('Cumulative Probability')
plt.grid(True, linestyle='--', alpha=0.7)

# 3. Box Plot (for outlier detection)
plt.subplot(2, 2, 3)
plt.boxplot(y_pred_1_final, vert=False)
plt.title('Flood Probability Box Plot')
plt.xlabel('Flood Probability')
plt.grid(True, linestyle='--', alpha=0.7)

# 4. Density Plot
plt.subplot(2, 2, 4)
sns.kdeplot(y_pred_1_final.flatten(), fill=True)
plt.title('Flood Probability Density Distribution')
plt.xlabel('Flood Probability')
plt.ylabel('Density')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('flood_probability_analysis.png')
print("Advanced analysis plots have been saved to flood_probability_analysis.png")
# 改进的正态性检验（抽样进行）
sample_size = min(5000, len(y_pred_1_final))
np.random.seed(42)
sample_indices = np.random.choice(len(y_pred_1_final), sample_size, replace=False)
y_sample = y_pred_1_final[sample_indices]

print("\n=== 分布特性分析 ===")
# === 分布特性分析 ===
print(f"平均值: {np.mean(y_pred_1_final):.4f}")
print(f"中位数: {np.median(y_pred_1_final):.4f}")
print(f"标准差: {np.std(y_pred_1_final):.4f}")

# 修复偏度和峰度计算
skewness = stats.skew(y_pred_1_final).item()  # 转为标量
kurt = stats.kurtosis(y_pred_1_final).item()  # 转为标量
print(f"偏度: {skewness:.4f}")
print(f"峰度: {kurt:.4f}")

# Shapiro-Wilk检验（抽样样本）
print("\n=== 正态性检验 (抽样n={}) ===".format(sample_size))
stat, p = stats.shapiro(y_sample)
print(f"Shapiro-Wilk检验统计量: {stat:.6f}, p值: {p:.6f}")
if p > 0.05:
    print("数据可能服从正态分布。")
else:
    print("数据不服从正态分布。")

# QQ图（使用抽样样本）
plt.figure(figsize=(8, 6))
stats.probplot(y_sample.flatten(), dist="norm", plot=plt)
plt.title('QQ图 (抽样样本)')
plt.grid(True)
plt.savefig('qq_plot_sampled.png')
print("QQ图已保存到qq_plot_sampled.png")

# 分布拟合建议
print("\n=== 分布拟合建议 ===")
if stats.skew(y_pred_1_final) > 1:
    print("数据呈现高度正偏态，建议考虑对数正态分布或Gamma分布")
elif stats.skew(y_pred_1_final) < -1:
    print("数据呈现高度负偏态，建议考虑Beta分布或Weibull分布")
else:
    print("数据偏度接近正态，但峰度分析显示{}".format(
        "尖峰分布" if stats.kurtosis(y_pred_1_final) > 0 else "平峰分布"
    ))