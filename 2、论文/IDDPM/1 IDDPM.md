
>论文：Improved denoising diffusion probabilistic models. 2021.

之间工作存在的问题：
- DDPM与其它概率生成模型，如VAE(Auto-encoding variational bayes.)、自回归模型(Conditional image generation with pixelcnn decoders)相比，效果不佳。
- DDPM显示出能在简单数据集，如Cifar10、LSUN上取得较好的效果，但是没有显示在更多种类的数据集，如ImageNet上取得较好的结果。
- Chen et al.(2020b, Wavegrad: Estimating gradients for waveform generation)展示出在音频领域，DDPM采样过程能够以较少的步骤生成音频，但是在图像领域没有这样的结果。


贡献：
- 





提到的其它论文的结论：
- Generating diverse high-fidelity images with vq-vae-2. 2019：optimizing log-likelihood forces generative models to capture all of the modes of the data distribution.
- Scaling laws for autoregressive generative modeling.2020：small improvements in log-likelihood can have a dramatic impact on sample quality and learnt feature representations.
