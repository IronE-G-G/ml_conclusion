# 提升方法

提升方法实际采用加法模型（基函数的线性组合）和向前分步算法。

## AdaBoost算法

AdaBoost算法是损失函数为指数损失的向前分步算法，只用于二分类。

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

<div align=center>

![function](http://latex.codecogs.com/gif.latex?f(x)=\sum^{M}_{m=1}\alpha_mG_m(x))  
</div>

所有![function](http://latex.codecogs.com/gif.latex?\alpha_m) 的和并不为1。
<div align=center>

![function](http://latex.codecogs.com/gif.latex?G(x)=sign(f(x)))  
</div>

## GBDT算法

**原理：** 每次在前一棵决策树的残差上构建下一棵决策树，将所有树线性组合。

### 向前分步算法

确定初始提升树为
$$f_0(x)=0 $$
第m步的模型为
$$ f_m(x) = f_{m-1}(x)+T(x;\theta_m)$$
其中，$f_{m-1}(x)$为当前模型，通过经验风险极小化确定下一棵决策树的$\theta_m$。
$$ \theta_m=\argmin\sum^{N}_{i=1}L(y_i, f_{m-1}(x_i)+T(x_i;\theta_m))$$
树的参数求解过程包括了特征选择过程以及叶子节点的权重求解，也就是树的生成了。

对于使用平方损失函数的回归问题，下一棵树的构建都是拟合当前的加法模型的残差。对于一般损失函数使用损失函数的负梯度，
$-[\frac{\partial{L(y,f(x_i))}}{\partial{f(x_i)}}]$  作为回归问题提升树算法中的残差的近似值。
