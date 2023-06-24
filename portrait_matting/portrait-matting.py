# BSHM人像抠图
# pip install opencv-python

# 导入cv2
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

# 转入模型
portrait_matting = pipeline(Tasks.portrait_matting,model='damo/cv_unet_image-matting')

# 传入图像
result = portrait_matting('./img/1.png')

# 返回结果
cv2.imwrite('./img/result.png', result[OutputKeys.OUTPUT_IMG])