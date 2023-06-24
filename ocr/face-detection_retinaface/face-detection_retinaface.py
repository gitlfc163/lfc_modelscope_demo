
# RetinaFace人脸检测关键点模型

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

retina_face_detection = pipeline(Tasks.face_detection, 'damo/cv_resnet50_face-detection_retinaface')
img_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/retina_face_detection.jpg'
result = retina_face_detection(img_path)
print(f'face detection output: {result}.')