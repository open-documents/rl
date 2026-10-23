
github：https://github.com/Stability-AI/stablediffusion


# stable-diffusion-2-base

huggingface：https://huggingface.co/stabilityai/stable-diffusion-2-base

使用了固定的、预训练的text encoder（OpenCLIP-ViT/H，github：https://github.com/mlfoundations/open_clip）。

环境准备：
```bash 
$ pip install diffusers transformers accelerate scipy safetensors
```
使用：
```python
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch

model_id = "stabilityai/stable-diffusion-2-base"

# Use the Euler scheduler here instead
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]  
    
image.save("astronaut_rides_horse.png")
```
注意：
- 尽管不是必需的依赖，出于内存和性能的考量，推荐安装 [xformers](https://github.com/facebookresearch/xformers)。
- 

