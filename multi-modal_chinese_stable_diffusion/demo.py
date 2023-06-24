
# 中文StableDiffusion-文本生成图像-通用领域
# 中文Stable Diffusion文生图模型, 输入描述文本，返回符合文本描述的2D图像。

# pip install diffusers
# pip install accelerate

import torch
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

task = Tasks.text_to_image_synthesis
model_id = 'damo/multi-modal_chinese_stable_diffusion_v1.0'


pipe = pipeline(task=task, model=model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
output = pipe({'text': '中国山水画', 'num_inference_steps': 50, 'guidance_scale': 7.5, 'negative_prompt':'模糊的'})
cv2.imwrite('result.png', output['output_imgs'][0])
