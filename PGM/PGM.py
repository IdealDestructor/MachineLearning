
# coding: utf-8

# #  概率图模型
# ## 1 基本概念
# 
# 概率图是一类用图的形式表示随机变量之间条件依赖关系的概率模型， 是概率论与图论的结合。图中的节点表示随机变量，缺少边表示条件独立假设。根据图中边的有向、无向性，模型可分为两类：有向图、无向图。
# 
# G(V,E)：变量关系图
# 
# V：顶点（节点），表示随机变量
# 
# E：边or弧
# 
# 两个节点邻接：两个节点之间存在边，记为$X_i\sim X_j$ ，不存在边，表示条件独立。
# 
# 路径：若对每个$i$，都有$X_{i-1}\sim X_i$ ，则称序列$（X_1, X_2 ... X_N ）$是一条路径。

# ## 2 几种常见的概率图模型
# 
# * 朴素贝叶斯分类器（NBs：Naive Bayes）
# * 最大熵模型（MEM：Maximum Entropy Model）
# * 隐马尔可夫模型（HMM：Hidden Markov Models）
# * 最大熵马尔可夫模型（MEMM：Maximum Entropy Markov Model）
# * 马尔可夫随机场（MRF：Markov Random Fields）
# * 条件随机场（CRF：Conditional Random Fields）

# ### 朴素贝叶斯分类器(NBs)
# #### 贝叶斯定理
# 在之前的实验中我们已经接触过简单的贝叶斯分类器了，现在再回顾一次：
# $$ P(y_i | x) = \frac{P(x|y_i)P(y_i)}{P(x)}$$
# 一般来说，$x$已给出，$P(x)$也是一个定值（虽然不知道准确的数据，但因为是恒值，可以忽略），只需关注分子$P(x|y_i)P(y_i)$。$P(y_i)$是类别$y_i$的先验概率，$P(x|y_i)$是$x$对类别$y_i$的条件概率。
# 贝叶斯定理说明了可以用先验概率$P(y_i)$来估算后验概率$P(x|y_i)$。
# 
# #### 贝叶斯分类器
# 设$x∈Ω$是一个类别未知的数据样本，$Y$为类别集合，若数据样本$x$属于一个特定的类别，那么分类问题就是决定$P(y_i|x)$，即在获得数据样本$x$时，确定$x$的最佳分类。所谓最佳分类，一种办法是把它定义为在给定数据集中不同类别$y_i$先验概率的条件下最可能的分类。贝叶斯理论提供了计算这种可能性的一种直接方法。
# 
# 举一个简单的例子:
# 
# $y_i$ 是一个包含了整数的数据集合$y_i=(1,1,1,2,2,5,...,86)$,每个$y_i$中的数据数量不一定相同，一共有$N$个这样的$y_i$数据集合，最终组成了一个拥有整数集合的数组。把这个数组当成已经划分好的不同类别。现在给出一个整数，比如1，问这个1属于哪一个集合或者说由某个类别yi产生该整数的可能性是多少？
#     
# 利用以上的贝叶斯定理可知，给定整数1的条件下，问属于$y_i$类别，就等同于求解先验概率$P(y_i)$与$P(x|y_i)$的概率乘积大小。$P(y_i)$表示类别$y_i$的分布概率，在这里可以简单地定义为$\frac{每个类别{y_i}的数据量}{总数据量}$（这种定义是有意义的，某个类别包含数据量越大，那么产生这个数据的可能性就越大）。另外，除了这个先验概率$P(y_i)$之外，还要考虑条件概率$P(x|y_i)$。在这个例子中，不同的$y_i$类别可能都包含了1这个整数，但是每个类别中1出现的概率不一样。所以，最后
# $$1属于y_i类别的概率 = 类别y_i发生的概率 * 1在类别yi中的出现概率$$
# 
# #### 贝叶斯网络(Bayesian Network)
# 贝叶斯网络是最基本的有向图，是类条件概率的建模方法。贝叶斯网络包括两部分：网络拓扑图和概率表。贝叶斯拓扑图的有向边指定了样本之间的关联。
# 
# ![fig1](./Img/fig1.png)
# 
# 每个节点的条件概率分布表示为：$$P(当前节点|它的父节点)$$
# 
# 联合分布为：
# $$P(X_1,X_2,...,X_N)=\prod_{i=1}^N p(X_i | \pi(X_i))$$
# 
# > 举例
# ![fig2](./Img/fig2.png)
# 联合分布为： 
# $$P(X_1,X_2,...,X_5)=  p(X_1)p(X_2 | X_1)p(X_3 | X_2) p(X_4 | X_2) p(X_5 | X_3 X_4)$$

# ###  最大熵分类器(MEM)
# 
# 最大熵模型主要是在已有的一些限制条件下估计未知的概率分布。最大熵的原理认为，从不完整的信息（例如有限数量的训练数据）推导出的唯一合理的概率分布应该在满足这些信息提供的约束条件下拥有最大熵值。求解这样的分布是一个典型的约束优化问题。
# ![fig3](./Img/fig3.png)
# 最大熵最后的模型公式——指数形式为：
# $$p_\lambda^* (y|x) = \frac{1}{z_\lambda (x)} exp(\sum_\alpha \lambda_\alpha f_\alpha(x , y))$$
# 
# 其中$z_\lambda (x)$是归一化因子， 最大熵模型公式中的$\lambda_\alpha$表示特征函数的权重，$ f_\alpha$可由训练样本估计得到，大的非负数值表示了优先选择的特征，大的负值对应不太可能发生的特征。

# ###  隐马尔可夫模型（HMM）
# 
# 状态集合$Y$，观察值集合$X$，两个状态转移概率：从$y_{i-1}$到$y_i$的条件概率分布$P(y_i | y_{i-1})$，状态$y_i$的输出观察值概率$P(x_i | y_{i-1})$，初始概率$P_0(y)$。
# ![fig4](./Img/fig4.png)
# 状态序列和观察序列的联合概率:
# $$P(\overrightarrow{y},\overrightarrow{x})=\prod_{i=1}^n p(y_i | y_{i-1})p(x_i | y_i)$$

# ###  最大熵马尔可夫模型（MEMM）
# 用一个分布$P(y_i | y_{i-1},x_i)$来替代HMM中的两个条件概率分布，它表示从先前状态$y_{i-1}$，在观察值$x_i$下得到当前状态的概率，即根据前一状态和当前观察预测当前状态。每个这样的分布函数都是一个服从最大熵的指数模型。
# 
# ![fig5](./Img/fig5.png)
# 状态$y_i$ 的条件概率公式（每个$i$ 的状态输出都服从最大熵的指数模型）
# $$p_{y_{i-1}} (y_i|x_i) = \frac{1}{z(x_i,y_{i-1})} exp(\sum_\alpha \lambda_\alpha f_\alpha(x_i , y_i))\ \ \ i = 1,2,...,T$$

# ###  马尔可夫随机场（MRF）
# 
# 随机场可以看成是一组随机变量$(y_1, y_2, …, y_n)$的集合（这组随机变量对应同一个样本空间）。当然，这些随机变量之间可能有依赖关系，一般来说，也只有当这些变量之间有依赖关系的时候，我们将其单独拿出来看成一个随机场才有实际意义。
# 
# 马尔可夫随机场是加了马尔可夫性限制的随机场，一个Markov随机场对应一个无向图。定义无向图G=(V,E)，V为顶点/节点, E为边，每一个节点对应一个随机变量，节点之间的边表示节点对应的随机变量之间有概率依赖关系。
# 
# **马尔可夫性**：
# 
# 对Markov随机场中的任何一个随机变量，给定场中其他所有变量下该变量的分布，等同于给定场中该变量的邻居节点下该变量的分布。即：
# $$P(y_i|y_{G/i})=P(y_i|N(y_i))$$
# 
# 其中$N(y_i)$表示与$y_i$有边相连的节点。
# 
# Markov随机场的结构本质上反应了我们的先验知识——哪些变量之间有依赖关系需要考虑，而哪些可以忽略。
# 
# 马尔可夫性可以看成是马尔科夫随机场的微观属性，而宏观属性就是联合分布。假设MRF的变量集合为$Y=\{ y_1, y_2,…, y_n \}$, $C_G$有是所有团$Y_c$的集合。
# $$P(y_1, y_2, ..., y_n)=\frac{1}{Z}exp\{-\frac{1}{T}U\{y_1, y_2, ... , y_n\}\}$$
# 
# 在MRF对应的图中，每一个团(clique)对应一个函数，称为团势(clique-potential)。这个联合概率形式又叫做Gibbs分布(Gibbs distribution)。
# 
# Hammersley-Clifford定理给出了Gibbs分布与MRF等价的条件：一个随机场是关于邻域系统的MRF，当且仅当这个随机场是关于邻域系统的Gibbs分布。关于邻域系
# 统$δ(s)$的MRFX与Gibbs分布等价形式表示为
# 
# $$P(y_s|y_r, r\in δ(s))=\frac{1}{Z}exp(-\frac{1}{T}\sum_{C \in C_G}V_c (y_s|y_r))$$
# 
# 在图像处理中，对先验模型的研究往往转换为对能量函数的研究。$C$表示邻域系统$δ$ 所包含基团的集合，$V_c (·)$是定义在基团$c$上的势函数(potential)，它只依赖于$δ(s)，s∈c$的值。$δ={δ(s)|s∈S}$是定义在S上的通用的邻域系统的集合。
# 
# 上式解决了求MRF中概率分布的难题，使对MRF的研究转化为对势函数Vc(x)的研究，使Gibbs分布与能量函数建立了等价关系，是研究邻域系统 $δ(s)$ MRF的一个重要里程碑。

# ###   条件随机场（CRF）
# 如果给定的MRF中每个随机变量下面还有观察值，我们要确定的是给定观察集合下MRF的分布，也就是条件分布，那么这个MRF就称为CRF(Conditional Random Field)。它的条件分布形式完全类似于MRF的分布形式，只不过多了一个观察集合$X=(x_1, x_2,…, x_n)$，即条件随机场可以看成是一个无向图模型或马尔可夫随机场，它是一种用来标记和切分序列化数据的统计模型。
# 
# 理论上，图G的结构可以任意，但实际上，在构造模型时，CRFs采用了最简单和最重要的一阶链式结构。
# 
# 一阶链式CRF示意图（不同于隐马尔科夫链，条件随机场中的$x_i$ 除了依赖于当前状态，还可能与其他状态有关）
# ![figx](./Img/figx.jpg)

# ## 3 几种概率图模型的比较
# ### 3.1 条件随机场和隐马尔科夫链的关系和比较
# 
# 条件随机场是隐马尔科夫链的一种扩展。
# 
# 不同点：观察值$x_i$不单纯地依赖于当前状态$y_i$，可能还与前后状态有关；
# 
# 相同点：条件随机场保留了状态序列的马尔科夫链属性——状态序列中的某一个状态只与之前的状态有关，而与其他状态无关。（比如句法分析中的句子成分）
# 
# ### 3.2 MRF和CRF的关系和比较
# 
# 条件随机场和马尔科夫随机场很相似，但有所不同，很容易混淆。最通用角度来看，CRF本质上是给定了观察值 (observations)集合的MRF。
#     
# 在图像处理中，MRF的密度概率 p(x=labels, y=image) 是一些随机变量定义在团上的函数因子分解。而CRF是根据特征产生的一个特殊MRF。因此一个MRF是由图和参数（可以无数个）定义的，如果这些参数是输入图像的一个函数（比如特征函数），则我们就拥有了一个CRF。
# 
# 图像去噪处理中，P(去噪像素|所有像素)是一个CRF，而P(所有像素)是一个MRF。[1]

# ## 4 代码实现
# 在Python库中，pgmpy实现了基本的概率图模型，大家可以在自己的终端执行下面的代码，完成pgmpy的安装
# ```
# pip install pgmpy
# 
# ```
# 这里我们将通过对下图所示的贝叶斯网络进行实现，说明pgmpy的用法。
# <img src="./Img/fig6.png" width = "300" height = "200" alt="cancer" align=center />
# 在上图所示的Bayes网络中，最上层表示导致癌症的因素污染和吸烟，中间层为癌症结点，最下层为癌症患者的可能发生的情况，分别是接受X光治疗和呼吸困难，下面利用pgmpy库实现上述贝叶斯网络，首先导入相应的模块：

# In[11]:


from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD


# 接下来定义相关参数：

# In[12]:


# 利用ByesianModel函数定义网络图
# 中括号内为整个图，小括号内前一项代表有向连接的起点，后一项代表终点

cancer_model = BayesianModel([('Pollution', 'Cancer'), 
                              ('Smoker', 'Cancer'),
                              ('Cancer', 'Xray'),
                              ('Cancer', 'Dyspnoea')])

# 利用TabularCPD函数赋予结点具体的概率参数
# cariable：参数名
# variable_card：该结点可能出现的结果数目
# values：概率值
cpd_poll = TabularCPD(variable='Pollution', variable_card=2,
                      values=[[0.9], [0.1]])
cpd_smoke = TabularCPD(variable='Smoker', variable_card=2,
                       values=[[0.3], [0.7]])

# evidence:可以理解为条件概率中的条件
cpd_cancer = TabularCPD(variable='Cancer', variable_card=2,
                        values=[[0.03, 0.05, 0.001, 0.02],
                                [0.97, 0.95, 0.999, 0.98]],
                        evidence=['Smoker', 'Pollution'],
                        evidence_card=[2, 2])
cpd_xray = TabularCPD(variable='Xray', variable_card=2,
                      values=[[0.9, 0.2], [0.1, 0.8]],
                      evidence=['Cancer'], evidence_card=[2])
cpd_dysp = TabularCPD(variable='Dyspnoea', variable_card=2,
                      values=[[0.65, 0.3], [0.35, 0.7]],
                      evidence=['Cancer'], evidence_card=[2])


# 这里的TabularCPD其实代表着条件概率分布表(Conditional Probability Distribution Table, CPDTable),我们可以看上述代码的cpd_cancer这个表，其实际的CPD表格为:
# 
# Smoker  | Smoker_0  | Smoker_0   | Smoker_1 | Smoker_1
# - | :-: | -:| -:
# Pollution  | Pollution_0  | Pollution_0   | Pollution_1 | Pollution_1
# Cancer_0 |  0.03        | 0.05        | 0.001       | 0.02        
# Cancer_1  | 0.97        | 0.95        | 0.999       | 0.98    
# 
# 以0.98所在的单元格为例，Smoker_1代表吸烟，Pullution_1代表有污染，Cancer_1代表患癌症，即在吸烟且环境有污染的情况下，患癌症概率为0.98。其他单元格阅读方式相同。这样，在知道任意情况的条件概率分布表的情况下，就能建立对应结点的参数。

# In[13]:


# 利用add_cpds函数将参数与图连接起来
cancer_model.add_cpds(cpd_poll, cpd_smoke, cpd_cancer, cpd_xray, cpd_dysp)

# 检查模型是否合理，True代表合理
cancer_model.check_model()


# In[14]:


# is_active_trail函数检验两个结点之间是否有有向连接
cancer_model.is_active_trail('Pollution', 'Smoker')


# In[15]:


# 在is_active_trail函数中，设置observed参数，表示两个结点能否通过observed结点实现连接
cancer_model.is_active_trail('Pollution', 'Smoker', observed=['Cancer'])


# # 5 实验练习
# 下面我们将利用一个更为复杂的Bayes网络，通过Pgmpy模块实现计算。网络图如下图：
# <img src="./Img/fig7.png" width = "500" height = "300" alt="Aisa" align=center />

# 首先导入相应的模块和数据集：
# >**注意:**pgmpy模块中，PGM图可以通过bif格式进行存储和阅读，这里已经将上述PGM以asia.bif储存好。

# In[16]:


from pgmpy.readwrite import BIFReader
from pgmpy.inference import VariableElimination
reader = BIFReader('./Img/asia.bif')
asia_model = reader.get_model()
# 通过nodes函数可以查看模型中有哪些结点
asia_model.nodes()


# >**练习1：**在下面的单元格中，实现判断，判断tub结点和either结点之间是否存在有向连接

# In[17]:


asia_model.is_active_trail('either', 'tub')


# >**练习2：**在下面的单元格中，实现判断，判断tub结点和dysp结点之间能否通过either结点有向连接

# In[18]:


asia_model.is_active_trail('tub', 'dysp', observed=['either'])


# 通过调用pgmpy的inference模块中的VariableElimination，我们可以实现查询条件概率表的功能

# In[19]:


asia_infer = VariableElimination(asia_model)
# 给出当smoke为0时，bronc的概率分布情况
q = asia_infer.query(variables=['bronc'], evidence={'smoke': 0})
print(q['bronc'])


# >**练习3：**在下面的单元格中，实现查询，当either为1时，xray的概率分布情况

# In[20]:


q = asia_infer.query(variables=['xray'], evidence={'either': 1})
print(q['xray'])


# # 引用和参考资料列表
# 1. [@概率图几种模型的简介和比较](http://blog.sina.com.cn/s/blog_60a0e97e0101no03.html)
# 2. [@技术博客](http://blog.csdn.net/pipisorry/article/details/51461878)
