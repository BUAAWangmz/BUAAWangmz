# 导入所需的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.stats import norm
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题


# 定义两个高斯分布的参数
m_A, sd_A = 164, 3  # 第一个高斯分布的均值和标准差
m_B, sd_B = 176, 5  # 第二个高斯分布的均值和标准差

# 从定义的两个高斯分布中生成样本
samp_A = np.random.normal(m_A, sd_A, 500)  # 从第一个高斯分布生成500个样本
samp_B = np.random.normal(m_B, sd_B, 1500) # 从第二个高斯分布生成1500个样本
samp_all = np.concatenate((samp_A, samp_B), axis=0)  # 将两组样本合并为一个数组

# 将合并后的样本数据保存到 CSV 文件中
height_data = pd.DataFrame(samp_all, columns=['height'])
height_data.to_csv('height_data.csv', index=False)

# 从 CSV 文件中读取身高数据，并将其转换为适合模型处理的格式
data = pd.read_csv('height_data.csv')
X = np.array(data['height']).reshape(-1, 1)

# 使用高斯混合模型（GMM）对数据进行建模
gmm = GaussianMixture(n_components=2)  # 创建一个包含两个高斯分布的 GMM 对象
gmm.fit(X)  # 使用身高数据训练 GMM

# 输出训练好的 GMM 的参数
print("均值:", gmm.means_)         # 均值
print("方差:", gmm.covariances_)  # 方差
print("权重:", gmm.weights_)       # 权重

# 插入新的代码
if gmm.means_[0][0] < gmm.means_[1][0]:
    gender1, gender2 = "女性", "男性"
else:
    gender1, gender2 = "男性", "女性"

print(f"{gender1} 均值: {gmm.means_[0][0]}, 方差: {gmm.covariances_[0][0][0]}, 权重: {gmm.weights_[0]}")
print(f"{gender2} 均值: {gmm.means_[1][0]}, 方差: {gmm.covariances_[1][0][0]}, 权重: {gmm.weights_[1]}")

# 使用训练好的 GMM 对身高数据进行预测
pred = gmm.predict(X)

# 使用轮廓系数来评估 GMM 的聚类质量
score = silhouette_score(X, pred)
print("Silhouette Score:", score)

# 计算训练好的 GMM 的对数似然值和贝叶斯信息准则（BIC）
log_likelihood = gmm.score(X)  # 对数似然值
bic_value = gmm.bic(X)         # BIC
print("对数似然值:", log_likelihood)
print("贝叶斯信息准则:", bic_value)

# 绘制身高数据的直方图
plt.figure()
plt.hist(data['height'], bins=20, density=True, alpha=0.6)

# 绘制拟合的高斯曲线
xmin, xmax = plt.xlim()  # 获取当前图形的 x 轴范围
x = np.linspace(xmin, xmax, 100)  # 创建一个在 x 轴范围内的等间距的点集
# 计算并绘制两个高斯分布的概率密度函数
p_A = norm.pdf(x, gmm.means_[0][0], np.sqrt(gmm.covariances_[0][0][0]))
p_B = norm.pdf(x, gmm.means_[1][0], np.sqrt(gmm.covariances_[1][0][0]))

# 根据权重绘制拟合的高斯曲线
plt.plot(x, p_A * gmm.weights_[0], 'r', linewidth=2)
plt.plot(x, p_B * gmm.weights_[1], 'r', linewidth=2)

# 设置图形中的轴标签和标题
plt.xlabel('身高 (厘米)', fontsize=14)         # x轴标签：身高（单位：厘米）
plt.ylabel('密度', fontsize=14)               # y轴标签：密度
plt.title('EM算法在混合高斯身高分析中的应用', fontsize=16)  # 图形标题



# 显示图形
plt.show()

