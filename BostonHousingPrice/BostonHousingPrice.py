#!/usr/bin/env python
# coding: utf-8

# # 房价预测--线性回归
# 
# 波士顿房价预测数据集是经典的机器学习、深度学习入门的数据集。下面我们用这个数据集完成房价预测任务。
# 
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/1c9a9b3b90b44dfdbba227098db93192aa52730d8fe647faad7c9fca1912e2c9" width="800" hegiht="" ></center>
# 
# 学习目标：  
# 1.了解深度学习框架编写代码的基本套路  
# 2.了解线性回归任务的基本模式

# In[34]:


#加载飞桨、Numpy和相关类库
import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph
from paddle.fluid.dygraph import Linear
import numpy as np
import os
import random


# 代码中参数含义如下：
# 
# * paddle/fluid：飞桨的主库，目前大部分的实用函数均在paddle.fluid包内。
# * dygraph：动态图的类库。
# * Linear：神经网络的全连接层函数，即包含所有输入权重相加和激活函数的基本神经元结构。在房价预测任务中，使用只有一层的神经网络（全连接层）来实现线性回归模型。
# <br></br>
# 
# ------
# 
# **说明：**
# 
# 飞桨支持两种深度学习建模编写方式，更方便调试的动态图模式和性能更好并便于部署的静态图模式。
# 
# * 静态图模式（声明式编程范式，类比C++）：先编译后执行的方式。用户需预先定义完整的网络结构，再对网络结构进行编译优化后，才能执行获得计算结果。
# * 动态图模式（命令式编程范式，类比Python）：解析式的执行方式。用户无需预先定义完整的网络结构，每写一行网络代码，即可同时获得计算结果。
# 
# 为了学习模型和调试的方便，本教程均使用动态图模式编写模型。在后续的资深教程中，会详细介绍静态图以及将动态图模型转成静态图的方法。仅在部分场景下需要模型转换，并且是相对容易的。
# 
# ------

# ## 数据处理
# 
# 数据处理包含五个部分：数据导入、数据形状变换、数据集划分、数据归一化处理和封装load data函数。数据预处理后，才能被模型调用。
# 数据处理的代码不依赖框架实现，直接使用 python 来完成任务
# 
# ### 数据形状变换
# 
# 由于读入的原始数据是1维的，所有数据都连在一起。因此需要我们将数据的形状进行变换，形成一个2维的矩阵，每行为一个数据样本（14个值），每个数据样本包含13个X（影响房价的特征）和一个Y（该类型房屋的均价）。
# 
# ### 数据集划分
# 
# 将数据集划分成训练集和测试集，其中训练集用于确定模型的参数，测试集用于评判模型的效果。为什么要对数据集进行拆分，而不能直接应用于模型训练呢？这与学生时代的授课和考试关系比较类似。
# 
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/a2c8c8cef82846678e3050b29d253aa6d1101eba0ceb4b099f42e3ad33d96928" width="600" hegiht="" ></center>
# <br></br>
# 
# 上学时总有一些自作聪明的同学，平时不认真学习，考试前临阵抱佛脚，将习题死记硬背下来，但是成绩往往并不好。因为学校期望学生掌握的是知识，而不仅仅是习题本身。另出新的考题，才能鼓励学生努力去掌握习题背后的原理。同样我们期望模型学习的是任务的本质规律，而不是训练数据本身，模型训练未使用的数据，才能更真实的评估模型的效果。
# 
# 在本案例中，我们将80%的数据用作训练集，20%用作测试集，实现代码如下。通过打印训练集的形状，可以发现共有404个样本，每个样本含有13个特征和1个预测值。
# 
# ### 数据归一化处理
# 
# 对每个特征进行归一化处理，使得每个特征的取值缩放到0~1之间。这样做有两个好处：一是模型训练更高效；二是特征前的权重大小可以代表该变量对预测结果的贡献度（因为每个特征值本身的范围相同）。

# In[35]:


def load_data():
    # 从文件导入数据
    datafile = 'work/data.txt'
    # data = np.fromfile(datafile, sep=',')
    data = np.loadtxt(datafile,delimiter=',')
    # 每条数据包括2项，其中前面1项是影响因素，第2项是相应的房屋价格中位数
    feature_names = [ 'AREA', 'MEDV' ]
    feature_num = len(feature_names)

    # 将原始数据进行Reshape，变成[N, 14]这样的形状
    # data = data.reshape([data.shape[0] // feature_num, feature_num])

    # 将原数据集拆分成训练集和测试集
    # 这里使用80%的数据做训练，20%的数据做测试
    # 测试集和训练集必须是没有交集的
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]

    # 计算train数据集的最大值，最小值，平均值
    maximums, minimums, avgs = training_data.max(axis=0), training_data.min(axis=0),                                  training_data.sum(axis=0) / training_data.shape[0]
    
    # 记录数据的归一化参数，在预测时对数据做归一化
    global max_values
    global min_values
    global avg_values
    max_values = maximums
    min_values = minimums
    avg_values = avgs

    # 对数据进行归一化处理
    for i in range(feature_num):
        #print(maximums[i], minimums[i], avgs[i])
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])

    # 训练集和测试集的划分比例
    #ratio = 0.8
    #offset = int(data.shape[0] * ratio)
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data


# ## 模型设计
# 
# 模型设计是深度学习模型关键要素之一，也称为网络结构设计，相当于模型的假设空间，即实现模型“前向计算”（从输入到输出）的过程。模型定义的实质是定义线性回归的网络结构，飞桨建议通过创建Python类的方式完成模型网络的定义，即定义``init``函数和``forward``函数。``forward``函数是框架指定实现前向计算逻辑的函数，程序在调用模型实例时会自动执行forward方法。在``forward``函数中使用的网络层需要在``init``函数中声明。
# 
# 实现过程分如下两步：
# 
# 1. **定义init函数**：在类的初始化函数中声明每一层网络的实现函数。在房价预测模型中，只需要定义一层全连接层
# 2. **定义forward函数**：构建神经网络结构，实现前向计算过程，并返回预测结果，在本任务中返回的是房价预测结果。
# 
# ### 线性回归模型设计
# 
# 如果将输入特征和输出预测值均以向量表示，输入特征$x$有13个分量，$y$有1个分量，那么参数权重的形状（shape）是$13\times1$。假设我们以如下任意数字赋值参数做初始化：
# $$w=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, -0.1, -0.2, -0.3, -0.4, 0.0]$$
# 
# 假设房价和各影响因素之间能够用线性关系来描述：
# 
# $$y = {\sum_{j=1}^Mx_j w_j} + b$$
# 
# 模型的求解即是通过数据拟合出每个$w_j$和$b$。其中，$w_j$和$b$分别表示该线性模型的权重和偏置。一维情况下，$w_j$ 和 $b$ 是直线的斜率和截距。
# 
# 线性回归模型使用均方误差作为损失函数（Loss），用以衡量预测房价和真实房价的差异，公式如下：
# 
# $$MSE = \frac{1}{n} \sum_{i=1}^n(\hat{Y_i} - {Y_i})^{2}$$

# 

# In[36]:


class Regressor(fluid.dygraph.Layer):
    def __init__(self):
        super(Regressor, self).__init__()
        
        # 定义一层全连接层，输出维度是1，激活函数为None，即不使用激活函数
        self.fc = Linear(input_dim=1, output_dim=1, act=None)
    
    # 网络的前向计算函数
    def forward(self, inputs):
        x = self.fc(inputs)
        return x


# ## 训练配置
# 
# 训练配置过程包含四步：
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/96075d4df5ae4e01ac1491ebf176fa557bd122b646ba49238f65c9b38a98cab4" width="700" hegiht="" ></center>
# <br></br>
# 
# 1. 以``guard``函数指定运行训练的机器资源，表明在``with``作用域下的程序均执行在本机的CPU资源上。``dygraph.guard``表示在``with``作用域下的程序会以飞桨动态图的模式执行（实时执行）。
# 1. 声明定义好的回归模型Regressor实例，并将模型的状态设置为训练。
# 1. 使用load_data函数加载训练数据和测试数据。
# 1. 设置优化算法和学习率，优化算法采用随机梯度下降[SGD](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/optimizer_cn/SGDOptimizer_cn.html#cn-api-fluid-optimizer-sgdoptimizer)，学习率设置为0.01。
# 
# 训练配置代码如下所示：

# In[37]:


# 定义飞桨动态图的工作环境
with fluid.dygraph.guard():
    # 声明定义好的线性回归模型
    model = Regressor()
    # 开启模型训练模式
    model.train()
    # 加载数据
    training_data, test_data = load_data() 
    # 定义优化算法，这里使用随机梯度下降-SGD
    # 学习率设置为0.01
    opt = fluid.optimizer.SGD(learning_rate=0.01, parameter_list=model.parameters())


# ------
# 
# **说明：**
# 
# 1. 默认本案例运行在读者的笔记本上，因此模型训练的机器资源为CPU。
# 1. 模型实例有两种状态：训练状态``.train()``和预测状态``.eval()``。训练时要执行正向计算和反向传播梯度两个过程，而预测时只需要执行正向计算。为模型指定运行状态，有两点原因：
# 
# （1）部分高级的算子（例如Drop out和Batch Normalization，在计算机视觉的章节会详细介绍）在两个状态执行的逻辑不同。
# 
# （2）从性能和存储空间的考虑，预测状态时更节省内存，性能更好。
# 
# 3. 在上述代码中可以发现声明模型、定义优化器等操作都在``with``创建的 [fluid.dygraph.guard()](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/dygraph_cn/guard_cn.html#guard)上下文环境中进行，可以理解为``with fluid.dygraph.guard()``创建了飞桨动态图的工作环境，在该环境下完成模型声明、数据转换及模型训练等操作。
# ------
# 

# ## 训练过程
# 
# 训练过程采用二层循环嵌套方式：
# 
# - **内层循环：** 负责整个数据集的一次遍历，采用分批次方式（batch）。假设数据集样本数量为1000，一个批次有10个样本，则遍历一次数据集的批次数量是1000/10=100，即内层循环需要执行100次。
# 
#         for iter_id, mini_batch in enumerate(mini_batches):
# 
# - **外层循环：** 定义遍历数据集的次数，通过参数EPOCH_NUM设置。
# 
#         for epoch_id in range(EPOCH_NUM):
# 
# ------
# **说明**:
# 
# batch的取值会影响模型训练效果。batch过大，会增大内存消耗和计算时间，且效果并不会明显提升；batch过小，每个batch的样本数据将没有统计意义。由于房价预测模型的训练数据集较小，我们将batch为设置10。
# 
# ------
# 
# 每次内层循环都需要执行如下四个步骤。
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/8154cf612a024a3f9144b4e31f59568ef9ad59c155b344919221d63bb9ccfcc8" width="700" hegiht="" ></center>
# <br></br>
# 
# 1. 数据准备：将一个批次的数据转变成np.array和内置格式。
# 1. 前向计算：将一个批次的样本数据灌入网络中，计算输出结果。
# 1. 计算损失函数：以前向计算结果和真实房价作为输入，通过损失函数[square_error_cost](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/layers_cn/square_error_cost_cn.html#square-error-cost/)计算出损失函数值（Loss）。飞桨所有的API接口都有完整的说明和使用案例。
# 1. 反向传播：执行梯度反向传播``backward``函数，即从后到前逐层计算每一层的梯度，并根据设置的优化算法更新参数``opt.minimize``。
# 
# 因为计算损失时需要把每个样本的损失都考虑到，所以我们需要对单个样本的损失函数进行求和，并除以样本总数$N$。
# $$L= \frac{1}{N}\sum_i{(y_i - z_i)^2}$$

# In[38]:


with dygraph.guard(fluid.CPUPlace()):
    EPOCH_NUM = 10   # 设置外层循环次数
    BATCH_SIZE = 10  # 设置batch大小
    
    # 定义外层循环
    for epoch_id in range(EPOCH_NUM):
        # 在每轮迭代开始之前，将训练数据的顺序随机的打乱
        np.random.shuffle(training_data)
        # 将训练数据进行拆分，每个batch包含10条数据
        mini_batches = [training_data[k:k+BATCH_SIZE] for k in range(0, len(training_data), BATCH_SIZE)]
        # 定义内层循环
        for iter_id, mini_batch in enumerate(mini_batches):
            x = np.array(mini_batch[:, :-1]).astype('float32') # 获得当前批次训练数据
            y = np.array(mini_batch[:, -1:]).astype('float32') # 获得当前批次训练标签（真实房价）
            # 将numpy数据转为飞桨动态图variable形式
            house_features = dygraph.to_variable(x)
            prices = dygraph.to_variable(y)
            
            # 前向计算
            predicts = model(house_features)
            
            # 计算损失
            loss = fluid.layers.square_error_cost(predicts, label=prices)
            avg_loss = fluid.layers.mean(loss)
            if iter_id%20==0:
                print("epoch: {}, iter: {}, loss is: {}".format(epoch_id, iter_id, avg_loss.numpy()))
            
            # 反向传播
            avg_loss.backward()
            # 最小化loss,更新参数
            opt.minimize(avg_loss)
            # 清除梯度
            model.clear_gradients()
    # 保存模型
    fluid.save_dygraph(model.state_dict(), 'LR_model')


# ## 保存并测试模型
# 
# ### 保存模型
# 
# 将模型当前的参数数据``model.state_dict()``保存到文件中（通过参数指定保存的文件名 LR_model），以备预测或校验的程序调用，代码如下所示。

# In[39]:


# 定义飞桨动态图工作环境
with fluid.dygraph.guard():
    # 保存模型参数，文件名为LR_model
    fluid.save_dygraph(model.state_dict(), 'LR_model')
    print("模型保存成功，模型参数保存在LR_model中")
    


# 理论而言，直接使用模型实例即可完成预测，而本教程中预测的方式为什么是先保存模型，再加载模型呢？这是因为在实际应用中，训练模型和使用模型往往是不同的场景。模型训练通常使用大量的线下服务器（不对外向企业的客户/用户提供在线服务），而模型预测则通常使用线上提供预测服务的服务器，或者将已经完成的预测模型嵌入手机或其他终端设备中使用。因此本教程的讲解方式更贴合真实场景的使用方法。
# 飞桨的愿景是用户只需要了解模型的逻辑概念，不需要关心实现细节，就能搭建强大的模型。
# 
# ### 测试模型
# 
# 下面我们选择一条数据样本，测试下模型的预测效果。测试过程和在应用场景中使用模型的过程一致，主要可分成如下三个步骤：
# 
# 1. 配置模型预测的机器资源。本案例默认使用本机，因此无需写代码指定。
# 1. 将训练好的模型参数加载到模型实例中。由两个语句完成，第一句是从文件中读取模型参数；第二句是将参数内容加载到模型。加载完毕后，需要将模型的状态调整为``eval()``（校验）。上文中提到，训练状态的模型需要同时支持前向计算和反向传导梯度，模型的实现较为臃肿，而校验和预测状态的模型只需要支持前向计算，模型的实现更加简单，性能更好。
# 1. 将待预测的样本特征输入到模型中，打印输出的预测结果。
# 
# 通过``load_one_example``函数实现从数据集中抽一条样本作为测试样本，具体实现代码如下所示。

# In[40]:


def load_one_example(data_dir):
    f = open(data_dir, 'r')
    datas = f.readlines()
    # 选择倒数第10条数据用于测试
    tmp = datas[-10]
    tmp = tmp.strip().split(', ')
    one_data = [float(v) for v in tmp]

    # 对数据进行归一化处理
    for i in range(len(one_data)-1):
        one_data[i] = (one_data[i] - avg_values[i]) / (max_values[i] - min_values[i])

    data = np.reshape(np.array(one_data[:-1]), [1, -1]).astype(np.float32)
    label = one_data[-1]
    return data, label


# In[41]:



with dygraph.guard():
    # 参数为保存模型参数的文件地址
    model_dict, _ = fluid.load_dygraph('LR_model')
    model.load_dict(model_dict)
    model.eval()

    # 参数为数据集的文件地址
    test_data, label = load_one_example('work/data.txt')
    # 将数据转为动态图的variable格式
    test_data = dygraph.to_variable(test_data)
    results = model(test_data)
 
    # 对结果做反归一化处理
    results = results * (max_values[-1] - min_values[-1]) + avg_values[-1]
    print("Inference result is {}, the corresponding label is {}".format(results.numpy(), label))


# 通过比较“模型预测值”和“真实房价”可见，模型的预测效果与真实房价接近。房价预测仅是一个最简单的模型，使用飞桨编写均可事半功倍。那么对于工业实践中更复杂的模型，使用飞桨节约的成本是不可估量的。同时飞桨针对很多应用场景和机器资源做了性能优化，在功能和性能上远强于自行编写的模型。
# 

# 

# In[ ]:




