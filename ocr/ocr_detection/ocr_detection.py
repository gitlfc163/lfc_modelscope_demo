
# 读光-文字检测-行检测模型-中英-通用领域--文本检测
# pip install tf_slim
# pip install pyclipper
# pip install shapely  

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import cv2
import numpy as np

# 载入模型
ocr_detection = pipeline(Tasks.ocr_detection, model='damo/cv_resnet18_ocr-detection-line-level_damo')
# ocr_detection = pipeline(Tasks.ocr_detection, model='damo/cv_resnet18_ocr-detection-word-level_damo')

# 使用图像文件
img_path = './img/test-invoice.png'
img = cv2.imread(img_path)
result = ocr_detection(img)

# 将numpy数组转换为列表，并按照多边形的中心点的横纵坐标之和进行排序，从小到大
det_result = result['polygons']
det_result_list = det_result.tolist()
det_result_list = sorted(det_result_list, key=lambda x: 0.01*sum(x[::2])/4+sum(x[1::2])/4) 
det_result_list2=np.array(det_result_list)

# 使用url
# result = ocr_detection('./img/test-invoice.png')
print(result['polygons'])   
print(det_result_list2)   