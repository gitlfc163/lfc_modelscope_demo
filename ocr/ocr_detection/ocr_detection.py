
# 读光-文字检测-行检测模型-中英-通用领域--文本检测
# pip install tf_slim
# pip install pyclipper
# pip install shapely  

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import cv2

# 载入模型
ocr_detection = pipeline(Tasks.ocr_detection, model='damo/cv_resnet18_ocr-detection-line-level_damo')
# ocr_detection = pipeline(Tasks.ocr_detection, model='damo/cv_resnet18_ocr-detection-word-level_damo')

# 使用图像文件
img_path = './img/test-invoice.png'
img = cv2.imread(img_path)
result = ocr_detection(img)

# 使用url
# result = ocr_detection('./img/test-invoice.png')
print(result)   