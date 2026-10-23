


```python
from diffusers import DDPMScheduler, UNet2DModel                                     

scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")                                      # 1
model = UNet2DModel.from_pretrained("google/ddpm-cat-256", use_safetensors=True).to("cuda")           # 2

scheduler.set_timesteps(50)                                                                           # 3

import torch

sample_size = model.config.sample_size
noise = torch.randn((1, 3, sample_size, sample_size)).to("cuda")
input = noise

for t in scheduler.timesteps:                                                                          #
	with torch.no_grad():                                                                              #
		noisy_residual = model(input, t).sample                                                        # 

	# 
	previous_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample    # previous_noisy_sample类型tensor
	input = previous_noisy_sample
```

下面对上面代码做逐行解释。

