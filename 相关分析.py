import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, pearsonr

# 设置中文字体，确保中文能正常显示
plt.rcParams["font.family"] = "SimHei"  # 只保留可用的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 尝试切换Matplotlib后端
import matplotlib
matplotlib.use('Agg')  # 使用Agg后端，不显示图形窗口，直接保存图片


def detect_encoding(file_path):
    """检测文件编码"""
    try:
        # 尝试使用chardet检测编码
        import chardet
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read(10000))
        return result['encoding']
    except ImportError:
        # 如果没有安装chardet，默认使用常见编码尝试
        for encoding in ['utf-8', 'gbk', 'gb2312', 'latin-1']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.readline()
                return encoding
            except UnicodeDecodeError:
                continue
        return 'utf-8'  # 默认返回utf-8


def load_data(file_path):
    """加载数据"""
    try:
        # 检测文件编码
        encoding = detect_encoding(file_path)
        print(f"检测到文件编码为: {encoding}")

        # 使用提供的表头
        headers = ['季风强度', '地形排水', '河流管理', '森林砍伐', '城市化', '气候变化', '大坝质量', '淤积',
                   '农业实践', '侵蚀', '无效防灾', '排水系统', '海岸脆弱性', '滑坡', '流域', '基础设施恶化', '人口得分',
                   '湿地损失', '规划不足', '政策因素', '洪水概率']

        # 读取数据，指定表头和编码
        df = pd.read_csv(file_path, encoding=encoding, header=None, names=headers)
        print(f"数据加载成功，共{df.shape[0]}行，{df.shape[1]}列")
        print(f"数据包含以下字段：{', '.join(df.columns.tolist())}")

        # 检查数据集行数是否异常大（如超过100万行）
        if df.shape[0] > 1000000:
            print("警告: 数据集行数异常大，可能是文件格式问题或包含重复数据")
            # 显示数据集前几行和后几行
            print("数据前几行预览:")
            print(df.head().to_csv(sep='\t', na_rep='nan'))
            print("数据后几行预览:")
            print(df.tail().to_csv(sep='\t', na_rep='nan'))

            # 询问用户是否继续
            choice = input("是否继续分析？(y/n): ")
            if choice.lower() != 'y':
                return None

        return df
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None


def preprocess_data(df):
    """数据预处理"""
    # 检查缺失值
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("检测到缺失值，处理中...")
        # 数值型列用中位数填充
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            # 如果没有识别出数值型列，尝试转换所有列
            print("警告: 未识别出数值型列，尝试转换所有列")
            for col in df.columns:
                if col != 'id':  # 跳过id列
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        print(f"无法将列 {col} 转换为数值类型")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        # 分类型列用众数填充
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])

    # 检查数据类型
    print("检查数据类型...")
    for col in df.columns:
        print(f"列 {col} 的数据类型: {df[col].dtype}")

    # 检查数值型列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("没有找到数值型列，无法进行相关性分析")

    # 检查异常值
    print("检查并处理异常值...")
    for col in numeric_cols:
        # 使用IQR方法检测异常值
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR

        # 替换异常值为上下限
        df[col] = df[col].clip(lower_bound, upper_bound)

    return df


def calculate_correlations(df, target_column='洪水概率'):
    """计算各种相关性"""
    # 确保目标列存在
    if target_column not in df.columns:
        raise ValueError(f"目标列 '{target_column}' 不存在于数据中")

    # 检查目标列是否包含有效数据
    if df[target_column].isnull().all():
        raise ValueError(f"目标列 '{target_column}' 全部为NaN值，无法进行相关性分析")

    # 提取数值型特征
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_features:
        numeric_features.remove(target_column)

    if not numeric_features:
        raise ValueError("没有找到数值型特征列，无法进行相关性分析")

    # 再次检查并处理NaN和inf值
    for col in numeric_features + [target_column]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(df[col].median())

    # 准备结果DataFrame
    correlations = pd.DataFrame(index=numeric_features,
                                columns=['皮尔逊相关系数', '斯皮尔曼相关系数', 'PCA权重', '综合得分'])

    # 1. 计算皮尔逊相关系数
    for feature in numeric_features:
        corr, p_value = pearsonr(df[feature], df[target_column])
        correlations.loc[feature, '皮尔逊相关系数'] = corr

    # 2. 计算斯皮尔曼相关系数
    for feature in numeric_features:
        corr, p_value = spearmanr(df[feature], df[target_column])
        correlations.loc[feature, '斯皮尔曼相关系数'] = corr

    # 3. 计算PCA权重
    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[numeric_features])

    # 执行PCA
    pca = PCA(n_components=len(numeric_features))
    pca.fit(X_scaled)

    # 计算每个特征在第一个主成分中的权重
    pca_weights = pd.Series(pca.components_[0], index=numeric_features, name='PCA权重')
    # 转换为正数以便于比较
    pca_weights = pca_weights.abs()
    # 归一化
    pca_weights = pca_weights / pca_weights.sum()

    correlations['PCA权重'] = pca_weights

    # 4. 计算综合得分（加权平均）
    # 为每种方法分配权重
    weights = {
        '皮尔逊相关系数': 0.4,  # 给予较高权重，因为线性关系直接相关
        '斯皮尔曼相关系数': 0.4,  # 考虑非线性关系
        'PCA权重': 0.2  # PCA提供了数据结构的信息
    }

    correlations['综合得分'] = (
            correlations['皮尔逊相关系数'] * weights['皮尔逊相关系数'] +
            correlations['斯皮尔曼相关系数'] * weights['斯皮尔曼相关系数'] +
            correlations['PCA权重'] * weights['PCA权重']
    )

    # 确保综合得分列是数值类型
    correlations['综合得分'] = pd.to_numeric(correlations['综合得分'], errors='coerce')

    # 按综合得分排序
    correlations = correlations.sort_values(by='综合得分', ascending=False)

    return correlations


def visualize_correlations(correlations, threshold=0.1):
    """可视化相关性分析结果"""
    plt.figure(figsize=(12, 10))

    # 检查综合得分列是否全为NaN
    if correlations['综合得分'].isnull().all():
        print("综合得分列全部为NaN值，无法绘制热力图")
        return None, None

    # 绘制热力图
    sns.heatmap(correlations[['综合得分']].sort_values(by='综合得分', ascending=False),
                annot=True, cmap='coolwarm', fmt='.3f', cbar=True, square=True)
    plt.title('各指标与洪水发生的相关性热力图')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')

    # 按相关性强度分类
    strong_correlations = correlations[correlations['综合得分'].abs() > threshold]
    weak_correlations = correlations[correlations['综合得分'].abs() <= threshold]

    print("\n与洪水发生密切相关的指标（综合得分绝对值 > {}）:".format(threshold))
    for index, row in strong_correlations.iterrows():
        print(f"- {index}: 综合得分 = {row['综合得分']:.3f}")

    print("\n与洪水发生相关性不大的指标（综合得分绝对值 <= {}）:".format(threshold))
    for index, row in weak_correlations.iterrows():
        print(f"- {index}: 综合得分 = {row['综合得分']:.3f}")

    # 绘制条形图 - 修复FutureWarning
    plt.figure(figsize=(14, 8))

    # 方法1：使用hue参数并禁用图例
    ax = sns.barplot(
        x=correlations.index,
        y='综合得分',
        data=correlations,
        hue=correlations.index,  # 将x变量同时指定给hue
        palette=sns.color_palette("coolwarm", len(correlations)),
        legend=False  # 禁用图例
    )

    # 为每个条形添加数值标签
    for i, v in enumerate(correlations['综合得分']):
        ax.text(i, v + 0.01 if v > 0 else v - 0.01, f"{v:.3f}", ha='center',
                color='black' if abs(v) > 0.05 else 'gray', fontweight='bold')

    plt.xticks(rotation=90)
    plt.title('各指标与洪水发生的相关性综合得分')
    plt.tight_layout()
    plt.savefig('correlation_scores.png', dpi=300, bbox_inches='tight')

    return strong_correlations, weak_correlations


def main():
    # 请替换为你的CSV文件路径
    file_path = 'train.csv'

    # 加载数据
    df = load_data(file_path)
    if df is None:
        return

    # 数据预处理
    df = preprocess_data(df)

    try:
        # 计算相关性
        correlations = calculate_correlations(df)

        # 可视化结果
        threshold = 0.1  # 定义相关性强弱的阈值，可以根据实际情况调整
        strong, weak = visualize_correlations(correlations, threshold)

        # 保存结果到CSV
        correlations.to_csv('correlation_results.csv', index=True)
        print("\n分析结果已保存至 'correlation_results.csv'")
    except ValueError as e:
        print(f"分析失败: {e}")


if __name__ == "__main__":
    main()