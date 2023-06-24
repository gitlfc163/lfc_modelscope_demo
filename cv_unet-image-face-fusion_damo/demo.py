
# 图像人脸融合

import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

image_face_fusion = pipeline(Tasks.image_face_fusion, model='damo/cv_unet-image-face-fusion_damo')
template_path = './img/facefusion_template.jpg'
user_path = './img/facefusion_user.jpg'
result = image_face_fusion(dict(template=template_path, user=user_path))

cv2.imwrite('./img/result3.png', result[OutputKeys.OUTPUT_IMG])
print('finished!')