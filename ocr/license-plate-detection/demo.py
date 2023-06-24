# 读光-车牌检测-通用

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
license_plate_detection = pipeline(Tasks.license_plate_detection, model='damo/cv_resnet18_license-plate-detection_damo')

img_url = './img/licenseplate_004.png'
result = license_plate_detection(img_url)

# result = license_plate_detection('https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/license_plate_detection.jpg')
print(result)
