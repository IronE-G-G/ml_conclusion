# GBDT的各种实现框架比较

作为GBDT我们的模型目标都是训练 K 个基模型然后将他们线性组合，即
<div align=center>

![function](http://latex.codecogs.com/gif.latex?\hat{y}_i=\sum^K_{k=1}f_k(x_i))  
</div>

## XGBoost

问题：

* 损失函数加入了模型复杂度，是不是算作预剪枝，还是说有另外的剪枝操作？  
* DART又是什么玩意？随机扔掉树避免过拟合？？

基模型用的是CART。  
假设我们已经训练好了前 t-1 个模型，对于第 t 个模型，我们需要最小化以下的目标函数：
<div align=center>

![function](http://latex.codecogs.com/gif.latex?L^{(t)}=\sum^N_{i=1}l(y_i,%20\hat{y}^{(t-1)}_i+f_t(x_i))+\Omega(f_t))  
![function](http://latex.codecogs.com/gif.latex?\Omega(f)=\tau%20T+\frac{1}{2}%20\lambda%20||\omega||^2)  
</div>

即这个损失函数是经验损失+模型复杂度惩罚的形式。其中对于模型复杂度的惩罚，除了对叶子数量的惩罚外，还多了一项对叶子权重的L2惩罚项，目的是避免过拟合(原文: The additional regularization term helps to smooth the final learnt weights to avoid over-fitting. )。  

对目标函数二阶泰勒展开：

令![function](http://latex.codecogs.com/gif.latex?f_t(x_i)=0)  

<div align=center>

![function](http://latex.codecogs.com/gif.latex?l(y_i,%20\hat{y}^{(t-1)}_i+f_t(x_i))\approx%20l(y_i,\hat{y}^{(t-1)}_i)+g_if_t(x_i)+\frac{1}{2}h_if^2_t(x_i))  

![function](http://latex.codecogs.com/gif.latex?g_i=\partial_{\hat{y}^{(t-1)}}l(y_i,\hat{y}^{(t-1)}),)  

![function](http://latex.codecogs.com/gif.latex?h_i=\partial^2_{\hat{y}^{(t-1)}}l(y_i,\hat{y}^{(t-1)}))  
</div>

复习一下泰勒展开公式：

<div align=center>

![function](http://latex.codecogs.com/gif.latex?f(x)=\sum^n_{i=1}\frac{f^{(i)}(x_0)}{i!}(x-x_0)^i)

</div>

因为在第 t 个基模型，前面t-1个模型组合的预测结果已知，去掉目标函数的常数项，
<div align=center>

![function](http://latex.codecogs.com/gif.latex?\tilde{L}^{(t)}=\sum^N_{i=1}[g_if_t(x_i)+\frac{1}{2}h_if^2_t(x_i)]+\Omega(f_t))

</div>



规定叶子 j 的样本集合为 ![function](http://latex.codecogs.com/gif.latex?I_j=\{i|q(x_i)=j\},%20q(x_i)) 将样本映射样本到所属的叶子节点。

<div align=center>

![function](http://latex.codecogs.com/gif.latex?\tilde{L}^{(t)}=\sum^N_{i=1}[g_if_t(x_i)+\frac{1}{2}h_if^2_t(x_i)]+\tau%20T+\frac{1}{2}\lambda\sum^T_{j=1}\omega^2_j)

![function](http://latex.codecogs.com/gif.latex?=\sum^T_{j=1}[(\sum_{i\in%20I_j}g_i)w_j+\frac{1}{2}(\sum_{i\in%20I_j}h_i+\lambda)w_j^2]+\tau%20T)
</div>

对$\omega_j$求偏导，让偏导结果为0，得到叶子 j 权重的最优解
$$ \omega^\star_j=-\frac{\sum_{i\in I_j}g_i}{\sum_{i\in I_j}h_i+\lambda}$$
然后代入计算得到的$\tilde{L}^{(t)}$可以作为模型评估的分数:
$$ \tilde{L}^{(t)}(q)=-\frac{1}{2}\sum^T_{i=1}\frac{(\sum_{i\in I_j}g_i)^2}{\sum_{i\in I_j}h_i+\lambda}+\tau T$$

在实际应用中，loss的下降程度最大也直接用作特征选择的准则。假设分割点分割数据集为$I=I_L\bigcup I_R$，
$$ \begin{aligned}
L_{split}&=L_{before}-L_{left}-L_{right}\\
&=-\frac{1}{2}[\frac{(\sum_{i\in I_L}g_i)^2}{\sum_{i\in I_L}h_i+\lambda}+\frac{(\sum_{i\in I_R}g_i)^2}{\sum_{i\in I_R}h_i+\lambda}-\frac{(\sum_{i\in I}g_i)^2}{\sum_{i\in I}h_i+\lambda}]+\tau
\end{aligned}
$$

### Shrinkage（收缩技术）

在每次学习到的模型的叶子权重乘一个学习率$\eta$，使得学习过程更加保守，能控制过拟合。
$$ y^{(m)}=y^{(m-1)}+\eta f_m(x_i), \eta \in (0,1]$$

### 列采样技术

列采样在控制过拟合上比行采样有效果。

根据python包提供的参数有按树/按层次/按节点进行列采样三个可以选择的参数，参数效果是叠加的。


### 模型参数

XGBoost有三种模型参数：

* 一般参数：主要是基模型的选择，通常是树模型或者线性模型；其他的有线程数，信息打印，是否disable掉默认的评估准则。
* booster相关参数（取决于使用的基模型）：主要分为整体控制树的结构，控制采样，控制节点分裂准则

  * learning_rate(default=0.3), shrinakge
  * min_split_loss(default=0), 分裂所需的最少损失降低
  * max_depth(default=6), 树深度
  * min_child_weight，分裂进行所需的最少的孩子节点权重，在线性回归任务中即孩子个数，在树模型中是hessian矩阵算
  * subsample(default=1), 在每一个基模型构建之前进行行采样，用采样的样本构建基模型。
  * sampling_method(default=uniform),还有一个是根据梯度采样，但因为计算量太大只在gpu下支持。
  * colsample_bytree, colsample_bylevel, colsample_bynode(default=1), 三个参数是累加效果。{'colsample_bytree':0.5, 'colsample_bylevel':0.5, 'colsample_bynode':0.5}在整体有64个特征的时候每一次split有8个特征可以选。
  * $lambda$(default=1), 叶子权重的L2惩罚项
  * $alpha$(default=0), 叶子权重的L1惩罚项
  * tree_method:支持exact(遍历全部取最优，贪婪算法)，approx(近似贪婪算法), hist(直方图算法)
  * scale_pos_weight(default=1)，缩放postive的样本的权重，如果使用的是auc做判断准则，那么这个参数对不平衡的样本集的训练就有帮助。
* 任务相关的参数：
  * objective(default=reg:squarederror) 目标函数
  * base_score(default=0.5)，最初的预测分数
  * eval_metric(默认值跟选择的目标函数相关)，训练集的评估准则。回归：rmse；分类：误差率；排序：平均准确率
  * seed(default=0)，随机种子

## LightGBM

## CatBoost
