# 决策树

**原理**：计算所有特征的信息增益（或其他），选择信息增益最大的节点分裂，直到信息增益很小或者叶子节点数目很少或者没有新的特征可以选。

决策树学习通常包括三个步骤：特征选择，决策树的生成，剪枝。

## ID3，C4.5，CART的区别

1. 节点分裂准则的不同，ID3使用信息增益，C4.5使用信息增益比，CART使用基尼指数。
2. CART能用于分类和回归，ID3，C4.5只能用于分类。
3. ID3和C4.5都只有树的生成，而CART包括了剪枝。

### 1. *熵(entropy)*

假设X是一个取有限个值的离散随机变量，其概率分布为

<div align=center>

![function](http://latex.codecogs.com/gif.latex?P(X=x_i)=p_i,%20i=1,2,...,n)  
</div>

则该随机变量的熵的表示为

<div align=center>

![function](http://latex.codecogs.com/gif.latex?H(x)=-\sum^{n}_{i=1}p_ilogp_i)  
</div>

### 2. *条件熵(conditional entropy)*

设有随机变量(X,Y)，其联合概率分布为

<div align=center>

![function](http://latex.codecogs.com/gif.latex?P(X=x_i,Y=y_i)=p_{ij},%20i=1,2,...,n;j=1,2...,m)  
</div>

给定![function](http://latex.codecogs.com/gif.latex?X)的条件下![function](http://latex.codecogs.com/gif.latex?Y)的条件熵为

<div align=center>

![function](http://latex.codecogs.com/gif.latex?H(Y|X)=\sum^{n}_{i=1}p_iH(Y|X=x_i))
</div>

条件熵是熵的加权平均。

### 3. *信息增益(information gain)*

特征![function](http://latex.codecogs.com/gif.latex?X)对训练目标![function](http://latex.codecogs.com/gif.latex?Y)的信息增益定义

<div align=center>

![function](http://latex.codecogs.com/gif.latex?g(Y,X)=H(Y)-H(Y|X))
</div>
使用信息增益存在偏向于选择去值较多的特征的问题，可以使用信息增益比较正。  
思考：一般在做模型训练之前会把多分类变量转成二分类的，是不是就能解决这个问题了。

### 4. *信息增益比(information gain ratio)*

在信息增益的基础熵除数据集关于特征X的值的熵：

<div align=center>

![function](http://latex.codecogs.com/gif.latex?g_R(Y,X)=\frac{g(Y,X)}{H_X(Y)})
</div>

其中，![function](http://latex.codecogs.com/gif.latex?H_X(Y)=-\sum^{n}_{i=1}\frac{|D_i|}{|D|}log_2\frac{|D_i|}{|D|}),![function](http://latex.codecogs.com/gif.latex?n)为特征![function](http://latex.codecogs.com/gif.latex?X)的取值个数。

### 5. *ID3，C4.5*

从根节点开始，对节点计算所有可能的特征的**信息增益**，选择最大的那个做分裂节点，对子节点递归调用以上方法，直到所有特征的信息增益均很小或者没有特征可以选择。

ID3算法只有树的生成，容易过拟合。C4.5在ID3上做了改进，用信息增益比来选择特征。

两种树的叶子节点的结果是选实例数最大的类做结果，所以只能用在分类上。  

### 6. *剪枝*

概念：在决策树中将已生成的树进行简化的过程叫做剪枝。决策树的剪枝往往通过极小化决策树整体的损失函数来实现。  
决策树的学习的损失函数可以定义为（可更改）

<div align=center>

![function](http://latex.codecogs.com/gif.latex?C_a(T)=\sum^{|T|}_{t=1}N_tH_t(T)+a|T|)
</div>

即经验损失部分+结构损失部分。
其中![function](http://latex.codecogs.com/gif.latex?H_t(T))是第 t 个节点的经验熵。
递归地将叶子节点往回缩，如果缩回去后的损失函数值变小，那么就往回缩。

### 7. *CART树*

CART算法包括二叉树的生成和剪枝，且可用于分类和回归问题。

对回归树采用的是**平方误差最小准则**，对分类树用**基尼指数最小化准则**，进行特征选择，生成二叉树。

算法停止的条件是节点中的样本个数小于阈值，或节点的基尼指数小于阈值，或没有更多特征可用来分裂节点。

叶子节点的结果回归取叶子节点中的所有实例的 Y 的均值，分类取多数表决。

### 8. *基尼指数*

分类问题中，假设有K个类，样本点属于第k类的概率为![function](http://latex.codecogs.com/gif.latex?p_k)，则概率分布的基尼指数为

<div align=center>

![function](http://latex.codecogs.com/gif.latex?Gini(p)=\sum^{K}_{k=1}p_k(1-p_k)=1-\sum^{K}_{k=1}p^2_k)
</div>

特征A将集合划分成两部分，在该条件下，集合D的基尼指数定义为

<div align=center>

![function](http://latex.codecogs.com/gif.latex?Gini(D,A)=\frac{|D_1|}{|D|}Gini(D_1)+\frac{|D_2|}{|D|}Gini(D_2))
</div>

跟熵类似，基尼指数越大，样本集合的不确定性也就越大。

### 9. *CART树的剪枝*

概括：剪枝，形成一个嵌套的子树序列；在剪枝得到的子树序列中通过交叉验证选取最优子树。

从整棵树![function](http://latex.codecogs.com/gif.latex?T_0)开始剪枝，对![function](http://latex.codecogs.com/gif.latex?T_0)的任意内部节点 t ，以 t 为单节点树的损失函数是

<div align=center>

![function](http://latex.codecogs.com/gif.latex?C_a(t)=C(T)+a)
</div>

以 t 为根节点的子树 ![function](http://latex.codecogs.com/gif.latex?T_t) 的损失函数为
<div align=center>

![function](http://latex.codecogs.com/gif.latex?C_a(T_t)=C(T_t)+a|T_t|)
</div>

当 a 充分小时有

<div align=center>

![function](http://latex.codecogs.com/gif.latex?C_a(T_t)<C_a(t))
</div>

当 a 增大，存在 a 使得
<div align=center>

![function](http://latex.codecogs.com/gif.latex?C_a(T_t)=C_a(t))
</div>
此时 a 的值为
<div align=center>

![function](http://latex.codecogs.com/gif.latex?a=\frac{C_a(t)-C_a(T_t)}{|T_t|-1})
</div>

只要 ![function](http://latex.codecogs.com/gif.latex?a=\frac{C_a(t)-C_a(T_t)}{|T_t|-1})，![function](http://latex.codecogs.com/gif.latex?T_t)跟 t 的损失函数值相同，而 t 的节点数少，那么 t 更可取。

***CART剪枝算法***

对决策树 T ，自下而上地对各内部节点t计算![function](http://latex.codecogs.com/gif.latex?C(T_t))，![function](http://latex.codecogs.com/gif.latex?|T_t|)以及
<div align=center>

![function](http://latex.codecogs.com/gif.latex?%20g(t)=\frac{C(t)-C(T_t)}{|T_t|-1},%20a%20=%20min(a,g(t)))
</div>

对![function](http://latex.codecogs.com/gif.latex?g(t)=a)的内部节点 t 进行剪枝，并对叶子节点 t 以多数表决法决定其类，得到树![function](http://latex.codecogs.com/gif.latex?T_1)。

如果![function](http://latex.codecogs.com/gif.latex?T_1)不是由根节点和两个叶子结点构成的树，重复以上步骤；

使用独立的验证数据集合，采用交叉验证法在子树序列![function](http://latex.codecogs.com/gif.latex?T_0,T_1,...,T_n)中选取最优子树![function](http://latex.codecogs.com/gif.latex?T_a)。

