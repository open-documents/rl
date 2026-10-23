
参考文档：
- `https://zhuanlan.zhihu.com/p/627442987`

模型估计的分布：$p(x_0)$（ 用来拟合现实中的分布）
现实中的分布：$q(x_0)$（

前向过程：$q(x_{0:T})$
$=q(x_{1:T}|x_0) \cdot q(x_0)$ 
$=q(x_{2:T}|x_0,x_1) \cdot q(x_1|x_0) \cdot q(x_0)$
$...$
$=q(x_0) \cdot q(x_1|x_0) \cdot q(x_2|x_0,x_1) \cdot q(x_3|x_0,x_1,x_2)...q(x_T|x_{0:T-1})$

由于前向过程假设为马尔可夫（Markov）过程，因此有$q(x_t|x_{t-1}) = q(x_t|x_{t-1},x_{t-2},...,x_0)$，

$=q(x_0) \cdot q(x_1|x_0) \cdot q(x_2|x_1) \cdot q(x_3|x_2)...q(x_{T-1}|x_{T-2}) \cdot q(x_T|x_{T-1})$
$=q(x_0)\prod_{t=1}^{T}q(x_t|x_{t-1})$



逆向（去噪）过程：$p(x_{0:T})$
$=p(x_{0:T-1}|x_T) \cdot p(x_T)$ 
$=p(x_{0:T-2}|x_{T-1},x_{T}) \cdot p(x_{T-1}|x_T) \cdot p(x_T)$
$...$
$=p(x_0|x_{1:T}) \cdot p(x_1|x_{2:T}) \cdot p(x_2|x_{3:T})...p(x_{T-1}|x_T) \cdot p(x_T)$

由于逆向过程假设为马尔可夫（Markov）过程，因此有 $p(x_t|x_{t+1}) = p(x_t|x_{t+1},x_{t+2},...,x_T)$，

$=p(x_0|x_1) \cdot p(x_1|x_2) \cdot p(x_2|x_3)...p(x_{T-2}|x_{T-1}) \cdot p(x_{T-1}|x_T) \cdot p(x_T)$
$=p(x_T)\prod_{t=1}^{T}p(x_{t-1}|x_{t})$


# 损失函数的推导


对估计的分布进行变换以方便建模。根据全概率公式：

$p(x_0)=\int p(x_{0:T})dx_{1:T}$

$=\int p(x_{0:T}) \frac{q(x_{1:T} | x_0)}{q(x_{1:T} | x_0)}dx_{1:T}$          

$=\int q(x_{1:T} | x_0) \frac{p(x_{0:T})}{q(x_{1:T} | x_0)}dx_{1:T}$  

$=\int q(x_{1:T} | x_0)p(x_T)\prod_{t=1}^{T}\frac{p(x_{t-1}|x_{t})}{q(x_t|x_{t-1})}dx_{1:T}$

从而损失函数：

$L=\int q(x_0)logp(x_0)dx_0$ （来自Deep Unsupervised Learning using Nonequilibrium Thermodynamics. 2015，个人理解就是交叉熵）（其中：q(x)为前向过程相关概率函数，即现实世界中的真实数据分布，p(x)为逆向过程相关概率函数，即估计分布）

$=\int q(x_0) \cdot log[\int q(x_{1:T} | x_0)p(x_T) \cdot \prod_{t=1}^{T}\frac{p(x_{t-1}|x_{t})}{q(x_t|x_{t-1})}dx_{1:T}]dx_0$

$\geq \int q(x_{0:T})\cdot log[p(x_T)\frac{p(x_{t-1}|x_{t})}{q(x_t|x_{t-1})}]d(x_{0:T}) = K$ 

从而目标函数可以变为优化

$K=E_{q(x_{0:T})}[ - \sum_{t=1}log(\frac{p(x_{t-1}|x_{t})}{q(x_t|x_{t-1})})-logp(x_T)]$

$=E_{q(x_{0:T})}[-log(\frac{p(x_0|x_1)}{q(x_1|x_0)})  - \sum_{t=2}log(\frac{p(x_{t-1}|x_{t})}{q(x_t|x_{t-1})})-logp(x_T)]$


-- -- 

对第二项的化简。记 B = $\sum_{t=2}log(\frac{p(x_{t-1}|x_{t})}{q(x_t|x_{t-1})})$

分母项：$q(x_t|x_{t-1})$

$=q(x_t|x_{t-1},x_0)$

$=\frac{q(x_t,x_{t-1},x_0)}{q(x_{t-1},x_0)}$

$=\frac{q(x_{t-1}|x_t,x_0)}{q(x_{t-1}|x_0)}\frac{q(x_t,x_0)}{q(x_0)}$

$=\frac{q(x_{t-1}|x_t,x_0)}{q(x_{t-1}|x_0)} q(x_t|x_0)$

从而

B = $\sum_{t=2}log(\frac{p(x_{t-1}|x_{t})q(x_{t-1}|x_0)}{q(x_{t-1}|x_t,x_0)q(x_t|x_0)})$

$=\sum_{t=2}log(\frac{p(x_{t-1}|x_{t})}{q(x_{t-1}|x_t,x_0)}) + \sum_{t=2}log(\frac{q(x_{t-1}|x_0)}{q(x_t|x_0)})$

$=\sum_{t=2}log(\frac{p(x_{t-1}|x_{t})}{q(x_{t-1}|x_t,x_0)}) + log(\prod_{t=2}\frac{q(x_{t-1}|x_0)}{q(x_t|x_0)})$

$=\sum_{t=2}log(\frac{p(x_{t-1}|x_{t})}{q(x_{t-1}|x_t,x_0)}) + log(\frac{q(x_{T-1}|x_0)}{q(x_T|x_0)}\frac{q(x_{T-2}|x_0)}{q(x_{T-1}|x_0)}\frac{q(x_{T-3}|x_0)}{q(x_{T-2}|x_0)})...\frac{q(x_2|x_0)}{q(x_3|x_0)}\frac{q(x_1|x_0)}{q(x_2|x_0)}$

$=\sum_{t=2}log(\frac{p(x_{t-1}|x_{t})}{q(x_{t-1}|x_t,x_0)}) + log(\frac{q(x_1|x_0)}{q(x_T|x_0)})$


-- --

K = $E_{q(x_{0:T})}[-log(\frac{p(x_0|x_1)}{q(x_1|x_0)})  - \sum_{t=2}log(\frac{p(x_{t-1}|x_{t})}{q(x_{t-1}|x_t,x_0)}) - log(\frac{q(x_1|x_0)}{q(x_T|x_0)})-logp(x_T)]$                                 （代入）

$=E_{q(x_{0:T})}[-log(\frac{p(x_0|x_1)}{q(x_T|x_0)})  - \sum_{t=2}log(\frac{p(x_{t-1}|x_{t})}{q(x_{t-1}|x_t,x_0)}) -logp(x_T)]$                                                          （第1项和第3项合并）

$=E_{q(x_{0:T})}[-log(p(x_0|x_1)  - \sum_{t=2}log(\frac{p(x_{t-1}|x_{t})}{q(x_{t-1}|x_t,x_0)}) -log\frac{p(x_T)}{q(x_T|x_0)}]$                                                       （第1项分母移到第3项中） 

$=E_{q(x_{0:T})}[-log(p(x_0|x_1)] + E_{q(x_{0:T})}[\sum_{t=2}-log(\frac{p(x_{t-1}|x_{t})}{q(x_{t-1}|x_t,x_0)})] + E_{q(x_{0:T})}[-log\frac{p(x_T)}{q(x_T|x_0)}]$


## 化简期望第二项

记 B = $E_{q(x_{0:T})}[\sum_{t=2}-log(\frac{p(x_{t-1}|x_{t})}{q(x_{t-1}|x_t,x_0)})]$

$=E_{q(x_{0:T})}[-log\frac{p(x_1|x_2)}{q(x_1|x_2,x_0)}] + E_{q(x_{0:T})}[-log\frac{p(x_2|x_3)}{q(x_2|x_3,x_0)}] + ... + E_{q(x_{0:T})}[-log\frac{p(x_{T-1}|x_{T})}{q(x_{T-1}|x_T,x_0)}]$   

$=E_{q(x_0,x_1,x_2)}[-log\frac{p(x_1|x_2)}{q(x_1|x_2,x_0)}] + E_{q(x_0,x_2,x_3)}[-log\frac{p(x_2|x_3)}{q(x_2|x_3,x_0)}] + ... + E_{q(x_0,x_{T-1},x_{T-2})}[-log\frac{p(x_{T-1}|x_{T})}{q(x_{T-1}|x_T,x_0)}]$          （对

$=\sum_{t=2} E_{q(x_0,x_{t-1},x_t)}[-log\frac{p(x_{t-1}|x_{t})}{q(x_{t-1}|x_t,x_0)}]$

$=\sum_{t=2} E_{q(x_t,x_0)}E_{q(x_{t-1}|x_t,x_0)}[-log\frac{p(x_{t-1}|x_{t})}{q(x_{t-1}|x_t,x_0)}]$

$=\sum_{t=2} E_{q(x_t,x_0)}D_{KL}(q(x_{t-1}|x_t,x_0)||p(x_{t-1}|x_{t}))$

$=\sum_{t=2} E_{q(x_T,...,x_{t+1},x_{t-1},...x_1)}E_{q(x_t,x_0)}D_{KL}(q(x_{t-1}|x_t,x_0)||p(x_{t-1}|x_{t}))$

$=\sum_{t=2} E_{q(x_{0:T})}D_{KL}(q(x_{t-1}|x_t,x_0)||p(x_{t-1}|x_{t}))$

$=E_{q(x_{0:T})}[\sum_{t=2} D_{KL}(q(x_{t-1}|x_t,x_0)||p(x_{t-1}|x_{t}))]$


对于针对 $E_{q(x_1,x_2)}f(y) = E_{q(x_1)}E_{q(x_2|x_1)}f(y) = E_{q(x_2)}E_{q(x_1|x_2)}f(y)$ 的证明（因为在积分中被积可以写在积分的任意位置，不受自变量的影响，所以以f(y)代替：

$E_{q(x_1,x_2)}f(y)$

$=\int\int q(x_1,x_2)f(y)dx_1dx_2$

$=\int\int q(x_1)q(x_2|x_1)f(y)dx_1dx_2$

$=\int q(x_1)dx_1[\int q(x_2|x_1)f(y)dx_2]$

$=\int q(x_1)[E_{q(x_2|x_1)}f(y)]dx_1$

$=E_{q(x_1)}[E_{q(x_2|x_1)}f(y)]$

$=E_{q(x_1)}E_{q(x_2|x_1)}f(y)$（写为）


-- --

## 化简期望第三项

C = $E_{q(x_{0:T})}[-log\frac{p(x_T)}{q(x_T|x_0)}]$

$=E_{q(x_0,x_T)}[-log\frac{p(x_T)}{q(x_T|x_0)}]$

$=E_{q(x_0,x_T)}[log\frac{q(x_T|x_0)}{p(x_T)}]$

$=E_{q(x_0)}E_{q(x_T|x_0)}[log\frac{q(x_T|x_0)}{p(x_T)}]$

$=E_{q(x_0)}[D_{KL}(q(x_T|x_0)||p(x_T))]$

$=E_{q(x_T,x_{T-1},...,x_1)}E_{q(x_0)}[D_{KL}(q(x_T|x_0)||p(x_T))]$

$=E_{q(x_{0:T})}[D_{KL}(q(x_T|x_0)||p(x_T))]$




从而

K = $E_{q(x_{0:T})}[-log(p(x_0|x_1)] + E_{q(x_{0:T})}[\sum_{t=2} D_{KL}(q(x_{t-1}|x_t,x_0)||p(x_{t-1}|x_{t}))] + E_{q(x_{0:T})}[D_{KL}(q(x_T|x_0)||p(x_T))]$




# Nice Property

已经假设前向过程中，$X_t \sim \mathcal N(\sqrt{1-\beta_t}X_{t-1},\beta_t\varepsilon)$，即：$q(x_t|x_{t-1})=\sqrt{1-\beta_t}x_{t-1}+\sqrt{\beta_t}\varepsilon$

下面定义：
- $\alpha_t=1-\beta_t$
- $\overline{\alpha}_t=\prod_{s=1}^{t}$

下面证明 $q(x_t|x_0) = $