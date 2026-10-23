
>Paper：Deconstructing Denoising Diffusion Models for Self-Supervised Learning.

>简记：Denoising Diffusion Models (DDM)、Denoising Autoencoder (DAE)

DDM中哪个/部分组件让DDM具有了学习表征的能力？

>Surprisingly, we discover that the main critical component is a tokenizer [33] that creates a low-dimensional latent space.

是创建低维的隐空间的Tokenizer让DDM具备了学习表征的能力。












# 相关论文

DDM相关文献：
- Deep unsupervised learning using nonequilibrium thermodynamics. ICML 2015
- Generative modeling by estimating gradients of the data distribution. NeurIPS 2019
- Score-based generative modeling through stochastic differential equations. 2020
- Denoising diffusion probabilistic models. NeurIPS, 2020
- Improved denoising diffusion probabilistic models. ICML 2021
- Diffusion models beat GANs on image synthesis. NeurIPS 2021

Scalable diffusion models with Transformers. ICCV 2023
High-resolution image synthesis with latent diffusion models. CVPR 2022

Extracting and composing robust features with denoising autoencoders. ICML 2008
- DAE最初提出来用于通过自监督的方式从数据中学习表征(representations)。
- 如今最成功的表征学习的DAE的变体都是基于“masking noise”，如自然语言领域中预测缺失文本的BERT、图像领域中预测缺失patch的MAE（Masked autoencoders are scalable vision learners. CVPR 2022）。

研究DDM学习表征的能力：
- Diffusion models beat GANs on image classification. 2023
- Denoising diffusion autoencoders are unified self-supervised learners. ICCV 2023
