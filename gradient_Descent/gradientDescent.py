import matplotlib
from mpl_toolkits.mplot3d import Axes3D

dataPath = r"./Input/data1.csv"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# # 梯度下降
# ## 1 算法简介
#
# >思考：我们给出一组房子面积，卧室数目以及对应房价数据，如何从数据中找到房价y与面积x<sub>1</sub>和卧室数目x<sub>2</sub>的关系？
#
# ![intropic](./Img/intro.png)
# 为了实现监督学习，我们选择采用自变量x1、x2的线性函数来评估因变量y值，得到：
# ![param1](./Img/param1.png)
# 在公式中$\theta$<sub>1</sub>和$\theta$<sub>2</sub>分别代表自变量$x$<sub>1</sub>和$x$<sub>2</sub>的权重(weights)，$\theta$<sub>0</sub>代表偏移量。为了方便，我们将评估值写作h(x)，令x<sub>0</sub>=1，则h(x)可以写作：
# ![param2](./Img/param2.png)
# 其中n为输入样本数的数量。为了得到权重的值，我们需要令我们目前的样本数据评估出的h(x)尽可能的接近真实y值。这里我们定义误差函数(cost function)来表示h(x)和y值相接近的程度：
# ![param3](./Img/param3.png)
# 这里的系数$\frac{1}{2}$是为了后面求解偏导数时可以与系数相互抵消。我们的目的是要误差函数尽可能的小，即求解权重使误差函数尽可能小。
#  <img src="./Img/pic3.png" width = "500" height = "500" alt="fig3" align=center />
# 如上图所示，只要自变量x沿着负梯度的方向变化，就可以到达函数的最小值了，反之，如果沿着正梯度方向变化，就可以到达函数的最大值。
# 我们要求解J函数的最小值，那么就要求出每个$\theta_{j}(j=0,1,2...n)$的梯度，由于梯度太大，可能会导致自变量沿着负梯度方向变化时，J的值出现震荡，而不是一直变小，所以在梯度的前面乘上一个很小的系数$\alpha$（学习率），对初始化的系数进行更新：
# ![param4](./Img/param4.png)
# 梯度计算公式（即偏导数）：
# ![param5](./Img/param5.png)
# 不断对系数进行更新，直至收敛（$\theta_{j}$的值几乎不发生变化），公式中m为数据样本的组数，i为第i组数据：
# ![algo1](./Img/algo1.png)
# 最后得到的$\theta_{j}$便是最终我们需要求解的线性方程的系数。

# ## 2 代码示例
# 首先先假设现在我们需要求解目标函数$func(x) = x * x$的极小值，由于func是一个凸函数，因此它唯一的极小值同时也是它的最小值，其一阶导函数为$dfunc(x) = 2 * x$

# In[1]:


# 目标函数:y=x^2
def func(x):
    return np.square(x)


# 目标函数一阶导数也即是偏导数:dy/dx=2*x
def dfunc(x):
    return 2 * x


# 接下来编写梯度下降法函数：

# In[2]:


# Gradient Descent
def GD(x_start, df, epochs, lr):
    """
    梯度下降法。给定起始点与目标函数的一阶导函数，求在epochs次迭代中x的更新值
    :param x_start: x的起始点
    :param df: 目标函数的一阶导函数
    :param epochs: 迭代周期
    :param lr: 学习率
    :return: x在每次迭代后的位置（包括起始点），长度为epochs+1
    """
    xs = np.zeros(epochs + 1)
    x = x_start
    xs[0] = x
    for i in range(epochs):
        dx = df(x)
        # v表示x要改变的幅度
        v = - dx * lr
        x += v
        xs[i + 1] = x
    return xs


# 在demo_GD中，我们直观地展示了如何利用梯度下降法的搜索过程：

# In[3]:


def demo_GD():
    # 演示如何使用梯度下降法GD()
    line_x = np.linspace(-5, 5, 100)
    line_y = func(line_x)

    x_start = -5
    epochs = 5

    lr = 0.3
    x = GD(x_start, dfunc, epochs, lr=lr)

    color = 'r'
    plt.plot(line_x, line_y, c='b')
    plt.plot(x, func(x), c=color, label='lr={}'.format(lr))
    plt.scatter(x, func(x), c=color, )
    plt.legend()
    plt.show()


demo_GD()

# 从运行结果来看，当学习率为0.3的时候，迭代5个周期似乎便能得到不错的结果了。
# > **思考：**在上述函数中，改变学习率，会对拟合的结果造成怎样的结果？请同学们尝试着将学习率(lr)改为0.1，0.5，0.9,观察上图的变化。

# # 3 练习题
# 回到我们之前的问题，给定数据集dataSet，每一行代表一组数据记录,每组数据记录中，第一个值为房屋面积（单位：平方英尺），第二个值为房屋中的房间数，第三个值为房价（单位：千美元），试用梯度下降法，构造损失函数，在函数gradientDescent中实现房价price关于房屋面积area和房间数rooms的线性回归，返回值为线性方程$price=\theta_0 + \theta_1 * area + \theta_2 * rooms$中系数$\theta_i(i=0,1,2)$的列表。

# In[416]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import genfromtxt
from mpl_toolkits.mplot3d import Axes3D

dataPath = r"./Input/data1.csv"
dataSet = pd.read_csv(dataPath, header=None, names=["area", "rooms", "price"])
print(dataSet)
# 数据初始化
m = len(dataSet)
origin_theta = np.array([[1], [0.5], [0.5]])
x0 = np.ones(m)
area = np.array((dataSet["area"]))
rooms = np.array((dataSet["rooms"]))
X = np.array([x0, area, rooms]).T
y = np.array(dataSet["price"]).reshape(len(dataSet), 1)
x1 = area
x2 = rooms
alpha = 0.0000002
epochs = 10
# 执行
# X, mu, sigma = normalizeFeatures(X)
theta, cost_history = gradientDescent(X, y, origin_theta, alpha, epochs)
printChart(theta, cost_history, x1, x2, y, X)
print(theta)


# In[49]:


def costFunction(X, y, theta):
    # 损失函数
    m = len(y)
    return ((X @ theta - y).T @ (X @ theta - y)) / (2 * m)


# In[42]:


def normalizeFeatures(X):
    # 归一化函数
    mu = np.zeros(len(X))
    sigma = np.zeros(len(X))
    for i, feature in enumerate(X.T):
        if i == 0: continue
        mu[i] = np.mean(feature)
        sigma[i] = np.std(feature)
        X[:, i] = ((feature - mu[i]) / sigma[i]).T
    return X, mu, sigma


# In[268]:


def gradientDescent(X, y, theta, alpha, epochs):
    # 梯度下降函数
    cost_history = []
    m = len(y)
    for i in range(epochs):
        theta = theta - (alpha / m) * X.T @ (X @ theta - y)
        cost_history.append(float(costFunction(X, y, theta)))
    return theta, cost_history


# In[234]:


def printChart(theta, cost_history, x1, x2, y, X):
    # 图表绘制函数
    plt.title('Cost Changes Tendency')
    plt.xlabel('epochs')
    plt.ylabel('cost')
    plt.plot(cost_history)

    fig = plt.figure()
    plt.title('Scatter plot of actual distribution & Fitting prediction surface')
    axe = plt.axes(projection='3d')
    axe.scatter3D(x1, x2, y, cmap='Pink')

    X, Y = np.meshgrid(x1, x2)
    Z = theta[0] + theta[1] * X + theta[2] * Y
    axe.plot_surface(X, Y, Z, cmap='rainbow')
