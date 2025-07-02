import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import time

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

# ===================== GPU设备设置 =====================
# 检查GPU可用性并设置设备[1,2,6](@ref)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")
# =====================================================

# 1. 数据加载（使用真实数据）
try:
    data = pd.read_csv("train.csv", encoding='gb18030')
    print("使用gb18030编码成功加载数据")
except UnicodeDecodeError:
    try:
        data = pd.read_csv("train.csv", encoding='utf-8-sig')
        print("使用utf-8-sig编码成功加载数据")
    except:
        import chardet

        with open("train.csv", 'rb') as f:
            result = chardet.detect(f.read())
        data = pd.read_csv("train.csv", encoding=result['encoding'])
        print(f"使用检测到的编码{result['encoding']}成功加载数据")

# 处理缺失值
if data.isnull().any().any():
    data = data.fillna(data.mean())
    print(f"填充了{data.isnull().sum().sum()}个缺失值")

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)
print(f"数据集准备完成: 特征维度 {X.shape}, 标签维度 {y.shape}")

# 2. 标准化处理
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)
print("数据标准化处理完成")

# 3. 数据集划分
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

train_size = int(0.7 * len(X_tensor))
val_size = int(0.15 * len(X_tensor))
test_size = len(X_tensor) - train_size - val_size

X_train, X_val, X_test = X_tensor[:train_size], X_tensor[train_size:train_size + val_size], X_tensor[
                                                                                            train_size + val_size:]
y_train, y_val, y_test = y_tensor[:train_size], y_tensor[train_size:train_size + val_size], y_tensor[
                                                                                            train_size + val_size:]

print(f"数据集划分: 训练集 {len(X_train)}, 验证集 {len(X_val)}, 测试集 {len(X_test)}")


# 4. 改进的神经网络模型
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
        print(f"模型初始化完成 - 输入维度: {input_dim}")

    def forward(self, x):
        return self.model(x)


# 5. 优化训练函数（GPU加速版）
def train_model(model, X_train, y_train, X_val, y_val, epochs=10, lr=0.001):
    print(f"\n开始训练模型...")
    print(f"训练参数: 总轮数 {epochs}, 学习率 {lr}")
    print(f"训练数据: {len(X_train)}个样本, 批大小 64")

    # ============= GPU关键修改1: 移动数据到GPU =============
    # 将验证集移动到GPU[1,2](@ref)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    # ===================================================

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # 创建DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    patience_limit = 15
    train_losses = []
    val_losses = []

    start_time = time.time()
    last_log_time = start_time

    for epoch in range(epochs):
        epoch_start_time = time.time()

        model.train()
        epoch_train_loss = 0
        batch_counter = 0

        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            # ============= GPU关键修改2: 移动batch数据到GPU =============
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            # ===========================================================

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_train_loss += loss.item() * batch_x.size(0)
            batch_counter += batch_x.size(0)

            # 每10个batch打印一次进度
            current_time = time.time()
            if current_time - last_log_time > 5 or epoch == 0 or epoch == epochs - 1:
                elapsed = current_time - start_time
                progress = (epoch + batch_idx / len(train_loader)) / epochs * 100
                print(f"Epoch {epoch + 1}/{epochs} | Batch {batch_idx + 1}/{len(train_loader)} "
                      f"| Progress: {progress:.1f}% | Loss: {loss.item():.6f} | "
                      f"Elapsed: {elapsed:.0f}s", end='\r')
                last_log_time = current_time

        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
            val_losses.append(val_loss)

        scheduler.step(val_loss)

        # 每个epoch都打印日志
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {val_loss:.6f} | "
              f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s     ")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"⏹️ 验证损失连续 {patience_limit} 轮未改善，触发早停!")
                break

    total_time = time.time() - start_time
    model.load_state_dict(best_model_state)
    print(f"✅ 训练完成! 总时间: {total_time:.0f}s, 最佳验证损失: {best_val_loss:.6f}")
    return train_losses, val_losses


# 6. 训练模型
print("\n" + "=" * 50)
print("开始模型训练流程")
print("=" * 50)
# ============= GPU关键修改3: 移动模型到GPU =============
model = FloodNetV3(input_dim=X_train.shape[1]).to(device)
# =====================================================
print(f"模型结构:\n{model}")

train_losses, val_losses = train_model(model, X_train, y_train, X_val, y_val, epochs=10)


# 7. 评估函数（使用R²计算，GPU兼容版）
def evaluate(model, X_test, y_test, scaler_y):
    print("\n开始模型评估...")
    model.eval()
    with torch.no_grad():
        # ============= GPU关键修改4: 移动测试数据到GPU =============
        X_test = X_test.to(device)
        y_test = y_test.to(device)
        preds = model(X_test)
        # 将结果移回CPU以便转换为numpy[2](@ref)
        preds = preds.cpu().numpy()
        y_true = y_test.cpu().numpy()
        # ========================================================

    # 反标准化
    preds_orig = scaler_y.inverse_transform(preds)
    y_true_orig = scaler_y.inverse_transform(y_true)

    # 计算指标
    mae = mean_absolute_error(y_true_orig, preds_orig)
    mse = mean_squared_error(y_true_orig, preds_orig)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_orig, preds_orig)

    print("\n" + "=" * 50)
    print("模型评估结果:")
    print("=" * 50)
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    # 可视化预测结果
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true_orig, preds_orig, alpha=0.3)
    plt.plot([y_true_orig.min(), y_true_orig.max()],
             [y_true_orig.min(), y_true_orig.max()], 'r--')
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('真实值 vs 预测值')
    plt.grid(True)
    plt.show()

    # 返回反标准化后的数据用于分析
    return mae, mse, r2, y_true_orig, preds_orig


# 8. 模型评估
mae, mse, r2, y_true, y_pred = evaluate(model, X_test, y_test, scaler_y)

# 9. 损失曲线可视化
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='TrainLoss')
plt.plot(val_losses, label='ValLoss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Vla Loss curve')
plt.legend()
plt.grid(True)
plt.show()

# 10. 模型保存
model_path = "flood_prediction_model.pth"
# 保存模型状态字典（设备无关）[5](@ref)
torch.save(model.state_dict(), model_path)
print(f"模型已保存至: {model_path}")

# 保存标准化器
joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")
print("数据标准化器已保存")

print("\n" + "=" * 50)
print("模型训练与评估流程完成!")
print("=" * 50)