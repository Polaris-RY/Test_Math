import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import boxcox
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 设置随机种子，确保结果可复现
np.random.seed(42)
torch.manual_seed(42)

# --------------------------- 1. 数据准备与预处理 --------------------------- #
# 读取真实数据（替换为你的数据路径）
try:
    data = pd.read_csv("train.csv")
    print(f"数据加载成功，共{len(data)}条记录，{len(data.columns)}个特征")
except:
    # 如果没有真实数据，生成模拟数据用于演示
    print("使用模拟数据进行演示...")
    data = pd.DataFrame(
        np.random.rand(100, 21),
        columns=[
            "海岸脆弱性", "侵蚀", "无防洪", "人口得分", "城市化", "流域", "森林砍伐",
            "规划不足", "地形排水", "淤积", "滑坡", "季风强度", "大坝质量",
            "基础设施恶化", "湿地损失", "河流管理", "农业实践", "气候变化",
            "政策因素", "排水系统", "洪水概率"  # 最后一列是目标
        ]
    )

# 数据预处理
# 检查并处理缺失值
if data.isnull().any().any():
    print("警告：数据包含缺失值，进行简单填充...")
    data = data.fillna(data.mean())

# 特征与目标分离
X = data.iloc[:, :-1].values  # 前20个指标特征
y = data.iloc[:, -1].values  # 洪水概率（或标签）
y = y.reshape(-1, 1)  # 适配网络输出维度

# 将二维数组转换为一维数组
y_1d = y.ravel()

# 对目标变量进行Box-Cox变换
y_boxcox, lambda_ = boxcox(y_1d + 1)  # 加1避免对0取对数
y_boxcox = y_boxcox.reshape(-1, 1)

# 标准化特征和目标
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y_boxcox)

import joblib

# 保存 scaler_y 和 lambda_
joblib.dump(scaler_y, 'scaler_y.pkl')
np.save('lambda_.npy', lambda_)

# 转为Tensor
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

# 划分训练集、验证集和测试集
train_size = int(0.7 * len(X_tensor))
val_size = int(0.15 * len(X_tensor))
test_size = len(X_tensor) - train_size - val_size

X_train, X_val, X_test = X_tensor[:train_size], X_tensor[train_size:train_size + val_size], X_tensor[train_size + val_size:]
y_train, y_val, y_test = y_tensor[:train_size], y_tensor[train_size:train_size + val_size], y_tensor[train_size + val_size:]

# --------------------------- 2. 定义神经网络模型 --------------------------- #
class FloodNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(FloodNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),  # 使用更稳定的激活函数
            nn.BatchNorm1d(hidden_dim),  # 添加批量归一化
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.layer(x)

# --------------------------- 3. 第一阶段：用相关性得分初始化权重并训练 --------------------------- #
# 加载相关性得分（对应20个指标的综合得分，需与特征顺序一致）
corr_scores = np.array([
    0.175, 0.170, 0.161, 0.160, 0.157, 0.156, 0.156, 0.155, 0.155, 0.153,
    0.153, 0.152, 0.151, 0.151, 0.149, 0.148, 0.148, 0.145, 0.144, 0.143
])

# 归一化相关性得分，作为第一层线性层的初始权重
corr_scores_norm = corr_scores / corr_scores.sum()

# 定义学习率列表
learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
best_lr = None
best_val_loss = float('inf')

for lr in learning_rates:
    # 初始化模型
    input_dim = 20
    model_1 = FloodNet(input_dim)

    # 改进的权重初始化方法
    with torch.no_grad():
        # 为每个隐藏层神经元创建不同的权重分布
        for i in range(model_1.layer[0].weight.shape[0]):
            # 使用权重得分的随机扰动，而不是完全相同的值
            perturbed_weights = corr_scores_norm * (1 + 0.1 * np.random.randn(len(corr_scores_norm)))
            perturbed_weights = np.abs(perturbed_weights)  # 确保权重非负
            perturbed_weights = perturbed_weights / perturbed_weights.sum()  # 重新归一化
            model_1.layer[0].weight[i] = torch.tensor(perturbed_weights, dtype=torch.float32)

        # 初始化偏置为小随机值
        model_1.layer[0].bias = nn.Parameter(torch.randn(model_1.layer[0].bias.shape) * 0.01)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_1.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)  # 学习率调度

    # 早停机制
    patience = 10
    counter = 0
    best_loss = float('inf')

    epochs = 100
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model_1.train()
        optimizer.zero_grad()
        outputs = model_1(X_train)
        loss = criterion(outputs, y_train)

        # 检查是否出现NaN
        if torch.isnan(loss):
            print(f"警告：第{epoch + 1}轮训练出现NaN损失！")
            print(f"输出检查：{outputs[:5]}")
            print(f"权重检查：{model_1.layer[0].weight[0][:5]}")
            break

        loss.backward()

        # 梯度裁剪，防止梯度爆炸
        nn.utils.clip_grad_norm_(model_1.parameters(), max_norm=1.0)

        optimizer.step()
        train_losses.append(loss.item())

        # 验证集损失
        model_1.eval()
        with torch.no_grad():
            val_outputs = model_1(X_val)
            val_loss = criterion(val_outputs, y_val).item()
            val_losses.append(val_loss)

        # 学习率调度
        scheduler.step(val_loss)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}, Val Loss: {val_loss:.6f}")

        # 早停检查
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if best_loss < best_val_loss:
        best_val_loss = best_loss
        best_lr = lr

print(f"Best learning rate for model 1: {best_lr}")

# 使用最佳学习率重新训练模型
input_dim = 20
model_1 = FloodNet(input_dim)

# 改进的权重初始化方法
with torch.no_grad():
    # 为每个隐藏层神经元创建不同的权重分布
    for i in range(model_1.layer[0].weight.shape[0]):
        # 使用权重得分的随机扰动，而不是完全相同的值
        perturbed_weights = corr_scores_norm * (1 + 0.1 * np.random.randn(len(corr_scores_norm)))
        perturbed_weights = np.abs(perturbed_weights)  # 确保权重非负
        perturbed_weights = perturbed_weights / perturbed_weights.sum()  # 重新归一化
        model_1.layer[0].weight[i] = torch.tensor(perturbed_weights, dtype=torch.float32)

    # 初始化偏置为小随机值
    model_1.layer[0].bias = nn.Parameter(torch.randn(model_1.layer[0].bias.shape) * 0.01)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model_1.parameters(), lr=best_lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)  # 学习率调度

epochs = 100
train_losses = []
for epoch in range(epochs):
    model_1.train()
    optimizer.zero_grad()
    outputs = model_1(X_train)
    loss = criterion(outputs, y_train)

    # 检查是否出现NaN
    if torch.isnan(loss):
        print(f"警告：第{epoch + 1}轮训练出现NaN损失！")
        print(f"输出检查：{outputs[:5]}")
        print(f"权重检查：{model_1.layer[0].weight[0][:5]}")
        break

    loss.backward()

    # 梯度裁剪，防止梯度爆炸
    nn.utils.clip_grad_norm_(model_1.parameters(), max_norm=1.0)

    optimizer.step()
    train_losses.append(loss.item())

    # 学习率调度
    scheduler.step(loss)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")

# 绘制训练损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('training_loss.png')

torch.save(model_1.state_dict(), 'model_1.pth')
print("模型 1 已保存为 model_1.pth")

# --------------------------- 4. 提取权重并筛选前五名指标 --------------------------- #
# 提取输入层->隐藏层的权重
input_weights = model_1.layer[0].weight.detach().numpy()  # 形状: [hidden_dim, 20]
# 计算每个指标的“平均权重”（绝对值的平均值，考虑正负影响）
avg_weights_per_feature = np.abs(input_weights).mean(axis=0)  # 形状: [20]

# 筛选前五名指标的索引和权重
top5_indices = np.argsort(avg_weights_per_feature)[-5:][::-1]  # 权重最大的5个指标索引
top5_weights = avg_weights_per_feature[top5_indices]
top5_weights_norm = top5_weights / top5_weights.sum()  # 归一化

# 打印筛选结果
feature_names = data.columns[:-1]
print("\n筛选出的权重最大的五个指标:")
for i, idx in enumerate(top5_indices):
    print(f"{i + 1}. 指标: {feature_names[idx]}, 权重: {avg_weights_per_feature[idx]:.4f}")

# --------------------------- 5. 第二阶段：用前五名权重初始化新网络并训练 --------------------------- #
# 检查 top5_indices 是否有重复值
if len(set(top5_indices)) != len(top5_indices):
    print("警告：top5_indices 中存在重复值，可能导致索引问题。")
    # 如果有重复值，可以尝试去重
    top5_indices = list(set(top5_indices))
    top5_indices.sort(reverse=True)  # 保持原来的顺序

# 复制 top5_indices 以避免负步长问题
top5_indices = top5_indices.copy()
top5_indices_tensor = torch.tensor(top5_indices, dtype=torch.long)

# 提取前五名指标的特征
try:
    X_top5 = X_tensor[:, top5_indices_tensor].clone()  # 创建副本，避免索引问题
except Exception as e:
    print(f"提取特征时出错：{e}")
    # 如果仍然出错，尝试对 X_tensor 进行复制
    X_top5 = X_tensor.clone()[:, top5_indices_tensor].clone()

# 划分训练集和测试集
X_train_top5, X_val_top5, X_test_top5 = X_top5[:train_size], X_top5[train_size:train_size + val_size], X_top5[train_size + val_size:]

# 为模型2寻找最佳学习率
best_lr_2 = None
best_val_loss_2 = float('inf')

for lr in learning_rates:
    # 定义新模型
    model_2 = FloodNet(input_dim=5)

    # 用筛选后的权重初始化新模型的输入层
    with torch.no_grad():
        # 为每个隐藏层神经元创建不同的权重分布
        for i in range(model_2.layer[0].weight.shape[0]):
            # 使用权重得分的随机扰动
            perturbed_weights = top5_weights_norm * (1 + 0.1 * np.random.randn(len(top5_weights_norm)))
            perturbed_weights = np.abs(perturbed_weights)  # 确保权重非负
            perturbed_weights = perturbed_weights / perturbed_weights.sum()  # 重新归一化
            model_2.layer[0].weight[i] = torch.tensor(perturbed_weights, dtype=torch.float32)

        # 初始化偏置为小随机值
        model_2.layer[0].bias = nn.Parameter(torch.randn(model_2.layer[0].bias.shape) * 0.01)

    # 定义损失函数和优化器
    optimizer_2 = optim.Adam(model_2.parameters(), lr=lr)
    scheduler_2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_2, 'min', patience=5, factor=0.5)

    # 早停机制
    patience = 10
    counter = 0
    best_loss_2 = float('inf')

    epochs = 100
    train_losses_2 = []
    val_losses_2 = []
    for epoch in range(epochs):
        model_2.train()
        optimizer_2.zero_grad()
        outputs = model_2(X_train_top5)
        loss = criterion(outputs, y_train)

        # 检查是否出现NaN
        if torch.isnan(loss):
            print(f"警告：模型2第{epoch + 1}轮训练出现NaN损失！")
            break

        loss.backward()
        nn.utils.clip_grad_norm_(model_2.parameters(), max_norm=1.0)
        optimizer_2.step()
        train_losses_2.append(loss.item())

        # 验证集损失
        model_2.eval()
        with torch.no_grad():
            val_outputs = model_2(X_val_top5)
            val_loss = criterion(val_outputs, y_val).item()
            val_losses_2.append(val_loss)

        # 学习率调度
        scheduler_2.step(val_loss)

        if (epoch + 1) % 20 == 0:
            print(f"Model 2 Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}, Val Loss: {val_loss:.6f}")

        # 早停检查
        if val_loss < best_loss_2:
            best_loss_2 = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Model 2 Early stopping at epoch {epoch + 1}")
                break

    if best_loss_2 < best_val_loss_2:
        best_val_loss_2 = best_loss_2
        best_lr_2 = lr

print(f"Best learning rate for model 2: {best_lr_2}")

# 使用最佳学习率重新训练模型2
model_2 = FloodNet(input_dim=5)

# 用筛选后的权重初始化新模型的输入层
with torch.no_grad():
    # 为每个隐藏层神经元创建不同的权重分布
    for i in range(model_2.layer[0].weight.shape[0]):
        # 使用权重得分的随机扰动
        perturbed_weights = top5_weights_norm * (1 + 0.1 * np.random.randn(len(top5_weights_norm)))
        perturbed_weights = np.abs(perturbed_weights)  # 确保权重非负
        perturbed_weights = perturbed_weights / perturbed_weights.sum()  # 重新归一化
        model_2.layer[0].weight[i] = torch.tensor(perturbed_weights, dtype=torch.float32)

    # 初始化偏置为小随机值
    model_2.layer[0].bias = nn.Parameter(torch.randn(model_2.layer[0].bias.shape) * 0.01)

# 定义损失函数和优化器
optimizer_2 = optim.Adam(model_2.parameters(), lr=best_lr_2)
scheduler_2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_2, 'min', patience=5, factor=0.5)

train_losses_2 = []
for epoch in range(epochs):
    model_2.train()
    optimizer_2.zero_grad()
    outputs = model_2(X_train_top5)
    loss = criterion(outputs, y_train)

    # 检查是否出现NaN
    if torch.isnan(loss):
        print(f"警告：模型2第{epoch + 1}轮训练出现NaN损失！")
        break

    loss.backward()
    nn.utils.clip_grad_norm_(model_2.parameters(), max_norm=1.0)
    optimizer_2.step()
    train_losses_2.append(loss.item())

    scheduler_2.step(loss)

    if (epoch + 1) % 20 == 0:
        print(f"Model 2 Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")

# 绘制训练损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses_2)
plt.title('Model 2 Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('training_loss_model2.png')

# --------------------------- 6. 模型评估 --------------------------- #
# 模型评估
model_1.eval()
model_2.eval()

with torch.no_grad():
    y_pred_1 = model_1(X_test)
    y_pred_2 = model_2(X_test_top5)

    mse_1 = criterion(y_pred_1, y_test).item()
    mse_2 = criterion(y_pred_2, y_test).item()

    print(f"\n模型1在测试集上的MSE: {mse_1:.6f}")
    print(f"模型2在测试集上的MSE: {mse_2:.6f}")

    # 计算R²分数
    y_test_np = y_test.numpy()
    y_pred_1_np = y_pred_1.numpy()
    y_pred_2_np = y_pred_2.numpy()

    r2_1 = 1 - np.sum((y_test_np - y_pred_1_np) ** 2) / np.sum((y_test_np - np.mean(y_test_np)) ** 2)
    r2_2 = 1 - np.sum((y_test_np - y_pred_2_np) ** 2) / np.sum((y_test_np - np.mean(y_test_np)) ** 2)

    print(f"模型1的R²分数: {r2_1:.6f}")
    print(f"模型2的R²分数: {r2_2:.6f}")

    mae_1 = mean_absolute_error(y_val.numpy(), y_pred_1.numpy())
    mae_2 = mean_absolute_error(y_val.numpy(), y_pred_2.numpy())

    print(f"模型1的mae分数:{mae_1:.6f}")
    print(f"模型2的mae分数:{mae_2:.6f}")