
# coding: utf-8

# # 奇异值分解(SVD)与主成分分析(PCA)
# ## 1 算法简介
# 
# 奇异值分解（Singular Value Decomposition），简称SVD，是线性代数中矩阵分解的方法。假如有一个矩阵A，对它进行奇异值分解，可以得到三个矩阵相乘的形式，最左边为m维的正交矩阵，中间为m*n 的对角阵，右边为n维的正交矩阵：
# $$A=U\Sigma V^{T}$$
# 这三个矩阵的大小如下图所示：
# ![fig1](./Img/fig1.png)
# 矩阵$\Sigma$除了对角元素其他元素都为0，并且对角元素是从大到小排列的，前面的元素比较大，后面的很多元素接近0。这些对角元素就是奇异值。($u_i$为m维行向量，$v_i$为n维行向量)
# <img src="./Img/fig3.png" width = "800" height = "600" alt="fig3" align=center />
# $\Sigma$中有n个奇异值，但是由于排在后面的很多接近0，所以我们可以仅保留比较大的前r个奇异值，同时对三个矩阵过滤后面的n-r个奇异值，
# <img src="./Img/fig4.png" width = "800" height = "600" alt="fig3" align=center />
# 奇异值过滤之后，得到新的矩阵：
# ![fig2](./Img/fig2.png)
# 在新的矩阵中，$\Sigma$只保留了前r个较大的特征值:
# <img src="./Img/fig5.png" width = "600" height = "400" alt="fig3" align=center />
# 
# 实际应用中，我们仅需保留三个比较小的矩阵，就能表示A，不仅节省存储量，在计算的时候更是减少了计算量。SVD在信息检索（隐性语义索引）、图像压缩、推荐系统、金融等领域都有应用。
# 
# 主成分分析（Principal Components Analysis），简称PCA，是一种数据降维技术，用于数据预处理。一般我们获取的原始数据维度都很高，比如1000个特征，在这1000个特征中可能包含了很多无用的信息或者噪声，真正有用的特征才100个，那么我们可以运用PCA算法将1000个特征降到100个特征。这样不仅可以去除无用的噪声，还能减少很大的计算量。
# 
# 简单来说，就是将数据从原始的空间中转换到新的特征空间中，例如原始的空间是三维的(x,y,z)，x、y、z分别是原始空间的三个基，我们可以通过某种方法，用新的坐标系(a,b,c)来表示原始的数据，那么a、b、c就是新的基，它们组成新的特征空间。在新的特征空间中，可能所有的数据在c上的投影都接近于0，即可以忽略，那么我们就可以直接用(a,b)来表示数据，这样数据就从三维的(x,y,z)降到了二维的(a,b)。
# <img src="./Img/fig6.png" width = "600" height = "400" alt="fig3" align=center />
# 
# 如上图所示，点在红线方向上分布最为集中，而在与红线方向垂直的方向上却鲜有分布，我们便可以将所有的蓝点投影在红线上，这样红线就成为了所有蓝点分布的主成分。原有的2维度特征降为1维度特征。一般步骤是这样的：先对原始数据零均值化，然后求协方差矩阵，接着对协方差矩阵求特征向量和特征值，这些特征向量组成了新的特征空间。
# 
# 在主成分分析中，特征值分解和奇异值分解都可以用来实现PCA。特征值和奇异值二者之间是有关系的：上面我们由矩阵A获得了奇异值$\Sigma_{i}$，假如方阵A\*A'的特征值为$\lambda_{i}$，则：$\Sigma_{i}^2=\lambda_{i}$。可以发现，求特征值必须要求矩阵是方阵，而求奇异值对任意矩阵都可以，因此PCA的实现其实用SVD的更多，在scikit-learn中，PCA算法其实也是通过SVD来实现的。[1]

# ## 2 代码实现
# 我们先实现SVD算法，首先导入所需要的模块：

# In[62]:


from numpy import *
from numpy import linalg as la


# 这里的数据集我们采用简单模拟的一个数据集，建立数据集载入函数LoadExData:

# In[63]:


# 加载测试数据集
# 数据矩阵的行对应用户，列对应物品
# 矩阵中第i行第j列表示第j个用户对第i个商品的评分
def loadExData():
    return mat([[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]])


# 接下来建立三种计算相似度的函数，分别用欧氏距离、皮尔逊相关系数和余弦相似度实现。同时归一化相似度：

# In[64]:


# 以下是三种计算相似度的算法，分别是欧式距离、皮尔逊相关系数和余弦相似度,
# 注意三种计算方式的参数inA和inB都是列向量
def ecludSim(inA,inB):
    return 1.0 / (1.0 + la.norm(inA - inB))  
    #范数的计算方法linalg.norm()，这里的1/(1+距离)表示将相似度的范围放在0与1之间

def pearsSim(inA,inB):
    if len(inA) < 3: 
        return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0][1]  
    # 皮尔逊相关系数的计算方法corrcoef()，
    # 参数rowvar=0表示对列求相似度，这里的0.5+0.5*corrcoef()是为了将范围归一化放到0和1之间

def cosSim(inA,inB):
    num = float(inA. T * inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num / denom) 
    #将相似度归一到0与1之间


# 利用函数sigmaPct确定需要保留的奇异值的个数k:

# In[65]:


# 按照前k个奇异值的平方和占总奇异值的平方和的百分比percentage来确定k的值,
# 后续计算SVD时需要将原始矩阵转换到k维空间
def sigmaPct(sigma,percentage):
    sigma2 = sigma ** 2 
    # 对sigma求平方
    sumsgm2 = sum(sigma2) 
    # 求所有奇异值sigma的平方和
    sumsgm3 = 0 
    #sumsgm3是前k个奇异值的平方和
    k=0
    for i in sigma:
        sumsgm3 +=  i ** 2
        k+=1
        if sumsgm3 >= sumsgm2 * percentage:
            return k


# 利用奇异值占比，确定预测评分：

# In[66]:


# 函数svdEst()的参数包含：数据矩阵、用户编号、物品编号和奇异值占比的阈值，
# 函数的作用是基于item的相似性对用户未评过分的物品进行预测评分
def svdEst(dataMat,user,simMeas,item,percentage):
    n=shape(dataMat)[1]
    simTotal=0.0;ratSimTotal=0.0
    u,sigma,vt=la.svd(dataMat)
    k=sigmaPct(sigma,percentage) 
    # 确定了需要保留的奇异值数量k
    sigmaK=mat(eye(k)*sigma[:k])  
    #构建对角矩阵
    xformedItems=dataMat.T*u[:,:k]*sigmaK.I  
    #根据k的值将原始数据转换到k维空间(低维),xformedItems表示物品(item)在k维空间转换后的值
    for j in range(n):
        userRating=dataMat[user,j]
        if userRating==0 or j==item:continue
        similarity=simMeas(xformedItems[item,:].T,xformedItems[j,:].T) 
        #计算物品item与物品j之间的相似度
        simTotal+=similarity 
        #对所有相似度求和
        ratSimTotal+=similarity*userRating 
        #用"物品item和物品j的相似度"乘以"用户对物品j的评分"，并求和
    if simTotal==0:return 0
    else:return ratSimTotal/simTotal 
    #得到对物品item的预测评分


# 最后我们利用余弦相似度的评价条件对测试数据集进行简单的推荐算法的实现：

# In[67]:


# 函数recommend()产生预测评分最高的N个推荐结果，默认返回5个；
# 参数包括：数据矩阵、用户编号、相似度衡量的方法、预测评分的方法、以及奇异值占比的阈值；
# 数据矩阵的行对应用户，列对应物品，函数的作用是基于item的相似性对用户未评过分的物品进行预测评分；
# 相似度衡量的方法默认用余弦相似度
def recommend(dataMat,user,N=5,simMeas=cosSim,estMethod=svdEst,percentage=0.9):
    unratedItems=nonzero(dataMat[user,:].A==0)[1]  
    #建立一个用户未评分item的列表
    if len(unratedItems)==0:
        return 'you rated everything' 
    #如果都已经评过分，则退出
    itemScores=[]
    for item in unratedItems:  
        #对于每个未评分的item，都计算其预测评分
        estimatedScore=estMethod(dataMat,user,simMeas,item,percentage)
        itemScores.append((item,estimatedScore))
    itemScores=sorted(itemScores,key=lambda x:x[1],reverse=True)
    #按照item的得分进行从大到小排序
    return itemScores[:N]  
    #返回前N大评分值的item名，及其预测评分值


# 这里我们对编号为1的用户推荐评分最高的三个商品，作为推荐算法的实现：

# In[68]:


testdata=loadExData()
print(recommend(testdata,1,N=3,percentage=0.8))
#对编号为1的用户推荐评分较高的3件商品


# 上述结果中，第一个数字代表商品编号，第二个数字代表评分。

# > **练习1: ** 在下面的代码框中，实现对编号为3的用户推荐评分最高的4件商品（代码填写在括号内）：

# In[69]:


print(recommend(testdata,3,N=4,percentage=0.8))


# > **练习2:**在下面的代码框中，修改某个参数，将计算相似度的方法改为欧氏距离、皮尔逊相关系数，实现对编号为1的用户推荐评分最高的3件商品，观察结果是否发生变化

# In[70]:


def recommend(dataMat,user,N=5,simMeas=cosSim,estMethod=svdEst,percentage=0.9):
    unratedItems=nonzero(dataMat[user,:].A==0)[1]  
    #建立一个用户未评分item的列表
    if len(unratedItems)==0:
        return 'you rated everything' 
    #如果都已经评过分，则退出
    itemScores=[]
    for item in unratedItems:  
        #对于每个未评分的item，都计算其预测评分
        estimatedScore=estMethod(dataMat,user,simMeas,item,percentage)
        itemScores.append((item,estimatedScore))
    itemScores=sorted(itemScores,key=lambda x:x[1],reverse=True)
    #按照item的得分进行从大到小排序
    return itemScores[:N]  
    #返回前N大评分值的item名，及其预测评分值
print("采用余弦相似度")
print(recommend(testdata,1,N=3,percentage=0.8))


# In[71]:


def recommend(dataMat,user,N=5,simMeas=ecludSim,estMethod=svdEst,percentage=0.9):
    unratedItems=nonzero(dataMat[user,:].A==0)[1]  
    #建立一个用户未评分item的列表
    if len(unratedItems)==0:
        return 'you rated everything' 
    #如果都已经评过分，则退出
    itemScores=[]
    for item in unratedItems:  
        #对于每个未评分的item，都计算其预测评分
        estimatedScore=estMethod(dataMat,user,simMeas,item,percentage)
        itemScores.append((item,estimatedScore))
    itemScores=sorted(itemScores,key=lambda x:x[1],reverse=True)
    #按照item的得分进行从大到小排序
    return itemScores[:N]  
    #返回前N大评分值的item名，及其预测评分值
print("采用欧式距离")
print(recommend(testdata,1,N=3,percentage=0.8))


# In[72]:


def recommend(dataMat,user,N=5,simMeas=pearsSim,estMethod=svdEst,percentage=0.9):
    unratedItems=nonzero(dataMat[user,:].A==0)[1]  
    #建立一个用户未评分item的列表
    if len(unratedItems)==0:
        return 'you rated everything' 
    #如果都已经评过分，则退出
    itemScores=[]
    for item in unratedItems:  
        #对于每个未评分的item，都计算其预测评分
        estimatedScore=estMethod(dataMat,user,simMeas,item,percentage)
        itemScores.append((item,estimatedScore))
    itemScores=sorted(itemScores,key=lambda x:x[1],reverse=True)
    #按照item的得分进行从大到小排序
    return itemScores[:N]  
    #返回前N大评分值的item名，及其预测评分值
print("采用皮尔逊相关系数")
print(recommend(testdata,1,N=3,percentage=0.8))


# 在PCA的实际应用中，我们一般调用scikit-learn中的PCA模块实现：

# In[73]:


from sklearn.decomposition import PCA 


# PCA对象主要有以下几种使用方法:
# 
# 
# * fit(X,y=None)
# 
# fit()可以说是scikit-learn中通用的方法，每个需要训练的算法都会有fit()方法，它其实就是算法中的“训练”这一步骤。因为PCA是无监督学习算法，此处y自然等于None。
# 
# fit(X)，表示用数据X来训练PCA模型。
# 
# 函数返回值：调用fit方法的对象本身。比如pca.fit(X)，表示用X对pca这个对象进行训练。
# 
# * fit_transform(X)
# 
# 用X来训练PCA模型，同时返回降维后的数据。
# newX=pca.fit_transform(X)，newX就是降维后的数据。
# 
# * inverse_transform()
# 
# 将降维后的数据转换成原始数据，X=pca.inverse_transform(newX)
# 
# * transform(X)
# 
# 将数据X转换成降维后的数据。当模型训练好后，对于新输入的数据，都可以用transform方法来降维。
# 
# 我们在这里导入一个随机生成的数据data：

# In[74]:


data = array(
      [[ 1.  ,  1.  ],  
       [ 0.9 ,  0.95],  
       [ 1.01,  1.03],  
       [ 2.  ,  2.  ],  
       [ 2.03,  2.06],  
       [ 1.98,  1.89],  
       [ 3.  ,  3.  ],  
       [ 3.03,  3.05],  
       [ 2.89,  3.1 ],  
       [ 4.  ,  4.  ],  
       [ 4.06,  4.02],  
       [ 3.97,  4.01]]) 


# data这组数据，有两个特征一共12个样本（x,y），其实就是分布在直线y=x上的点，并且聚集在x=1、2、3、4上，因为两个特征是近似相等的，所以用一个特征就能表示了

# In[79]:


pca=PCA(n_components=1)
newData=pca.fit_transform(data)
print(newData)


# 这里我们可以发现，PCA中有一个参数n_components，事实上PCA调用的时候一共可以设置三个参数：
# * n_components:  
# 
#     意义：PCA算法中所要保留的主成分个数n，也即保留下来的特征个数n
# 
#     类型：int 或者 string，缺省时默认为None，所有成分被保留。
#           
#           赋值为int，比如n_components=1，将把原始数据降到一个维度。
#           
#           赋值为string，比如n_components='mle'，将自动选取特征个数n，使得满足所要求的方差百分比。
# 
# * copy:
# 
#     类型：bool，True或者False，缺省时默认为True。
# 
#     意义：表示是否在运行算法时，将原始训练数据复制一份。若为True，则运行PCA算法后，原始训练数据的值不会有任何改变，因为是在原始数据的副本上进行运算；若为False，则运行PCA算法后，原始训练数据的值会改，因为是在原始数据上进行降维计算。
# 
# * whiten:
# 
#     类型：bool，缺省时默认为False
# 
#     意义：白化，使得每个特征具有相同的方差。关于“白化”，可参考[Ufldl教程](http://deeplearning.stanford.edu/wiki/index.php/%E7%99%BD%E5%8C%96)

# >**练习3：**利用mle方法对data进行PCA降维，观察结果的变化:

# In[78]:


pca=PCA(n_components='mle')
newDataByMle=pca.fit_transform(data)
print(newDataByMle)


# # 引用和参考资料列表
# 1.[@奇异值分解（SVD)](http://blog.csdn.net/u012162613/article/details/42214205)
# 
# 2.[@机器学习实战](http://item.jd.com/11242112.html)
# 
# 3.[@python 代码实现](http://www.cnblogs.com/lzllovesyl/p/5243370.html)
