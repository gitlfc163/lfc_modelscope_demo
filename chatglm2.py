from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline

import torch

pipe = pipeline(task=Tasks.chat, model='ZhipuAI/chatglm2-6b', model_revision='v1.0.2')
inputs = {'text':'你好', 'history': []}
result = pipe(inputs)
inputs = {'text':'介绍下清华大学', 'history': result['history']}
result = pipe(inputs)
print(result)