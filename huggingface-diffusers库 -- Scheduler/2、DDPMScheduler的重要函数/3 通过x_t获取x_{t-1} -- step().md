
在此之前，我们已经根据模型获得了 $\Large{\varepsilon_θ}$  



```python
def step(
	self,
	model_output: torch.FloatTensor,
	timestep: int,
	sample: torch.FloatTensor,
	generator=None,
	return_dict: bool = True,
	)
	-> Union[DDPMSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).
        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`.
        Returns:
            [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.
        """

	t = timestep
	# 计算(t-1), 以timestep=[980, 960, 940, ...]为例, t=980, (t-1)=960 
	prev_t = self.previous_timestep(t)      
	# 
	if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
		model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
	else:
		predicted_variance = None
	# 1. compute alphas, betas
	alpha_prod_t = self.alphas_cumprod[t]                                            # 
	alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one     # 
	beta_prod_t = 1 - alpha_prod_t                                                   # 
	beta_prod_t_prev = 1 - alpha_prod_t_prev                                         # 
	current_alpha_t = alpha_prod_t / alpha_prod_t_prev                               #
	current_beta_t = 1 - current_alpha_t                                             #  

	# 2. compute predicted original sample from predicted noise also called "predicted x_0" of formula (15) 
	# from https://arxiv.org/pdf/2006.11239.pdf
	if self.config.prediction_type == "epsilon":
		pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
	elif self.config.prediction_type == "sample":
		pred_original_sample = model_output
	elif self.config.prediction_type == "v_prediction":
		pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
	else:
		raise ValueError("...")

	# 3. Clip or threshold "predicted x_0"
	if self.config.thresholding:
		pred_original_sample = self._threshold_sample(pred_original_sample)
	elif self.config.clip_sample:
		pred_original_sample = pred_original_sample.clamp(
			-self.config.clip_sample_range, self.config.clip_sample_range
		)

	# 4. Compute coefficients(系数) for pred_original_sample x_0 and current sample x_t
	# See formula (7) from https://arxiv.org/pdf/2006.11239.pdf(DDPM)
	pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
	current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

	# 5. Compute predicted previous sample µ_t
	# See formula (7) from https://arxiv.org/pdf/2006.11239.pdf(DDPM)
	pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

	# 6. Add noise
	variance = 0
	if t > 0:
	    device = model_output.device
	    variance_noise = randn_tensor(
	        model_output.shape, generator=generator, device=device, dtype=model_output.dtype
	    )
	    if self.variance_type == "fixed_small_log":
	        variance = self._get_variance(t, predicted_variance=predicted_variance) * variance_noise
	    elif self.variance_type == "learned_range":
	        variance = self._get_variance(t, predicted_variance=predicted_variance)
	        variance = torch.exp(0.5 * variance) * variance_noise
	    else:
	        variance = (self._get_variance(t, predicted_variance=predicted_variance) ** 0.5) * variance_noise
	pred_prev_sample = pred_prev_sample + variance
	if not return_dict:
	    return (pred_prev_sample,)
	return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)
```

# 计算方差 

```python
def _get_variance(self, t, predicted_variance=None, variance_type=None):
	prev_t = self.previous_timestep(t)          # 计算x_{t-1}
	alpha_prod_t = self.alphas_cumprod[t]
	alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
	current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev
	# For t > 0, compute predicted variance βt (see formula (6) and (7) from DDPM)
	# and sample from it to get previous sample
	# x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
	variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
	# we always take the log of variance, so clamp it to ensure it's not 0
	variance = torch.clamp(variance, min=1e-20)
	if variance_type is None:
		variance_type = self.config.variance_type

	# hacks - were probably added for training stability
	if variance_type == "fixed_small":
		variance = variance
	# for rl-diffuser https://arxiv.org/abs/2205.09991(Planning with Diffusion for Flexible Behavior Synthesis)
	elif variance_type == "fixed_small_log":
		variance = torch.log(variance)
		variance = torch.exp(0.5 * variance)
	elif variance_type == "fixed_large":
		variance = current_beta_t
	elif variance_type == "fixed_large_log":
		# Glide max_log
		variance = torch.log(current_beta_t)
	elif variance_type == "learned":
		return predicted_variance
	elif variance_type == "learned_range":
		min_log = torch.log(variance)
		max_log = torch.log(current_beta_t)
		frac = (predicted_variance + 1) / 2
		variance = frac * max_log + (1 - frac) * min_log
	return variance
```