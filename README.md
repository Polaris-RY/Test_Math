# 洪水风险分析与预测项目

## 项目简介
本项目包含洪水风险相关性分析、聚类分析、深度学习预测及结果分析等模块，旨在对洪水发生概率进行数据驱动的建模与分析。

## 目录结构
- `p1_相关分析.py`：对输入数据进行相关性分析，输出各指标与洪水概率的相关性热力图和综合得分。
- `p2_聚类分析.py`：基于K-means算法对洪水概率进行聚类，分析不同风险类别的特征。
- `p3_prediction_new.py`：基于PyTorch的神经网络模型进行洪水概率预测，支持GPU加速，输出模型评估指标及训练曲线。
- `p4.py`：加载训练好的模型，对测试集进行预测，并输出概率分布分析及可视化图表。
- `train.csv`/`test.csv`：训练与测试数据文件（需自行准备）。
- `scaler_X.pkl`/`scaler_y.pkl`/`lambda_.npy`：模型训练过程中保存的标准化器和Box-Cox参数。
- `flood_prediction_model.pth`：训练好的模型权重文件。

## 环境依赖
- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy
- torch (PyTorch)
- joblib
- chardet（可选，用于自动检测编码）

建议使用`pip install -r requirements.txt`安装依赖（需自行整理requirements.txt）。

## 使用方法

1. **相关性分析**
   ```bash
   python p1_相关分析.py
   ```
   输出相关性分析结果及可视化图片。

2. **聚类分析**
   ```bash
   python p2_聚类分析.py
   ```
   输出聚类类别及可视化。

3. **模型训练与预测**
   ```bash
   python p3_prediction_new.py
   ```
   训练神经网络模型，保存模型和标准化器。

4. **模型推理与分布分析**
   ```bash
   python p4.py
   ```
   对测试集进行预测，输出概率分布分析和可视化图表。

## 注意事项
- 请确保`train.csv`和`test.csv`文件格式与脚本要求一致。
- 若遇到编码问题，可根据提示调整文件编码。
- 训练模型时建议使用GPU以加快速度。
