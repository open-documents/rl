论文：Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. arXiv:2006.11239, 2020.

贡献：
- 展示了DDPM和SM(Score Matching)之间的关系。
- 




# 附录

等式

$$
\begin{aligned}
log \ p_\theta(x_\theta) 
&= log \int p_\theta(x_{0:T}) \ dx_{1:T}  \\
&= log \int q(x_{1:T}|x_0) \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)} \ dx_{1:T}  \\
&= log \ E_{q(x_{1:T}|x_0)}  \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)} \\
&\ge E_{q(x_{1:T}|x_0)} \ log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)} \\
\Rightarrow -log \ p_\theta(x_\theta) & \le E_{q(x_{1:T}|x_0)} \ \big[ -log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)} \ \big]
\end{aligned}
$$


