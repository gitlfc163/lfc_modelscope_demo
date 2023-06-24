
# 读光-文字检测-行检测模型-中英-通用领域--文本识别
# pip install tf_slim
# pip install pyclipper
# pip install shapely  

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import cv2

# 载入模型
ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-general_damo')
# ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-scene_damo')
# ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-document_damo')
# ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-handwritten_damo')

# 使用图像文件
# img_path = './img/test-invoice.png'
# img = cv2.imread(img_path)
# result = ocr_recognition(img)

# 使用url
result = ocr_recognition('./img/test-invoice.png')
print(result)   