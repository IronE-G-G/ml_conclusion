# 提升方法

## AdaBoost算法

**原理**：提高那些被前一轮弱分类器分类错误的样本权重，降低那些被正确分类的样本权重，训练本轮分类器。将所有弱分类器加权组合起来形成一个强分类器。

* 初始化训练数据的权值，训练一个弱分类器，计算分类误差，根据分类误差计算这个分类器的权重；提高被分错的数据的权重，降低分类正确的数据的权重。再在新的权重分布上训练分类器；
* 结果是这些弱分类器的加权组合
* Loss：加权分类误差率

### 基分类器的训练

假设第 m 轮的训练数据权值分布为
<div align=center>

![function](http://latex.codecogs.com/gif.latex?D_m=(w_{m1},%20w_{m2},%20...,%20w_{mN}))  
</div>

学习该数据集，得到基分类器
<div align=center>

![function](http://latex.codecogs.com/gif.latex?G_m(x):%20\chi\rightarrow\{-1,1\})  
</div>

计算该分类器的分类误差率：
<div align=center>

![function](http://latex.codecogs.com/gif.latex?e_m=P(G_m(x_i)\not%20={y_i})=\sum^{N}_{i=1}w_{mi}I(G_m(x_i)\not%20={y_i}))  
</div>

计算该分类器的权重（用的是对数几率？？）：
<div align=center>

![function](http://latex.codecogs.com/gif.latex?\alpha_m=\frac{1}{2}log\frac{1-e_m}{e_m})  
</div>

更新下一轮数据集的权值分布（指数函数）：
<div align=center>

![function](http://latex.codecogs.com/gif.latex?w_{m+1,i}=\frac{w_{mi}}{Z_m}exp(-\alpha_my_iG_m(x_i)))  
</div>

其中 ![function](http://latex.codecogs.com/gif.latex?Z_m)  是规范化因子。

### 组合基分类器

$$ f(x)=\sum^{M}_{m=1}\alpha_mG_m(x)$$
所有$\alpha_m$的和并不为1。
$$ G(x)=sign(f(x))$$

## GBDT框架

**原理：** 每次在前一棵决策树的残差上构建下一棵决策树，将所有树的结果加起来。

gbdt和xgboost的区别：

* 正则化-对叶子节点个数做了惩罚，对叶子节点分数做惩罚。减少过拟合。
* 二阶泰勒展开，更接近loss函数

lightgbm和xgboost的区别：

* lightgbm基于梯度的单边采样（GOSS）
* xgboost采用预分类算法和直方图算法
* 直方图算法：将特征值按照大小分成不同区间；对于一个特征，pre-sorted 需要对每一个不同特征值都计算一次分割增益，而 histogram 只需要计算 #bin (histogram 的横轴的数量) 次。不能找到很精确的分割点
  
catboost和xgboost的区别：

* 对离散性特征处理做了改进，还有自动特征组合功能
* xgboost需要对类别特征做独热编码
