
# 读光-文字识别-行识别模型-中英-车牌文本领域

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import cv2

ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-licenseplate_damo')

### 使用url
img_url = './img/licenseplate_001.png'
result = ocr_recognition(img_url)
print(result)

### 使用图像文件
### 请准备好名为'ocr_recognition_licenseplate.jpg'的图像文件
# img_path = './img/001.png'
# img = cv2.imread(img_path)
# result = ocr_recognition(img)
# print(result)