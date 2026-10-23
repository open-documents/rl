
github地址：https://github.com/runwayml/stable-diffusion

包含下面6个ckpt：
- sd-v1-1.ckpt：
- sd-v1-2.ckpt：
- sd-v1-3.ckpt：
- sd-v1-4.ckpt：
- sd-v1-5.ckpt：

# stable-diffusion-v1-4

代码库（参考文档）：
- huggingface -- https://huggingface.co/CompVis/stable-diffusion-v1-4

**Stable-Diffusion-v1-4** checkpoint首先用stable-diffusion-v1-2的权重，然后

）模型描述

使用了imagen文章中使用的预训练的 text encoder CLIP-L/14
- imagen paper：https://arxiv.org/abs/2205.11487
- CLIP-L/14：Learning Transferable Visual Models From Natural Language Supervision

）模型使用

注意点：如果GPU显存小于4GB，确保按下面的方式以float16的精度加载模型，而不是默认的float32。
```python
import torch
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]  

image.save("astronaut_rides_horse.png")
```

改变Scheduler。
```python
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

model_id = "CompVis/stable-diffusion-v1-4"

# Use the Euler scheduler here instead
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]  
    
image.save("astronaut_rides_horse.png")

```


# stable-diffusion-v1-5


