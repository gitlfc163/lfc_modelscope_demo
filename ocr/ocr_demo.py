# 读光-文字检测-行检测模型-中英-通用领域--检测识别串联

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
# Python数值计算库
import numpy as np
import cv2
import math

# 对图像进行裁剪:scripts for crop images
# 参数接受一个图像和一个位置参数:img表示一个图像对象，position表示一个包含四个顶点坐标的列表，例如[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
def crop_image(img, position):
    def distance(x1,y1,x2,y2):
        return math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))    
    position = position.tolist()
    for i in range(4):
        for j in range(i+1, 4):
            if(position[i][0] > position[j][0]):
                tmp = position[j]
                position[j] = position[i]
                position[i] = tmp
    if position[0][1] > position[1][1]:
        tmp = position[0]
        position[0] = position[1]
        position[1] = tmp

    if position[2][1] > position[3][1]:
        tmp = position[2]
        position[2] = position[3]
        position[3] = tmp

    x1, y1 = position[0][0], position[0][1]
    x2, y2 = position[2][0], position[2][1]
    x3, y3 = position[3][0], position[3][1]
    x4, y4 = position[1][0], position[1][1]

    corners = np.zeros((4,2), np.float32)
    corners[0] = [x1, y1]
    corners[1] = [x2, y2]
    corners[2] = [x4, y4]
    corners[3] = [x3, y3]

    img_width = distance((x1+x4)/2, (y1+y4)/2, (x2+x3)/2, (y2+y3)/2)
    img_height = distance((x1+x2)/2, (y1+y2)/2, (x4+x3)/2, (y4+y3)/2)

    corners_trans = np.zeros((4,2), np.float32)
    corners_trans[0] = [0, 0]
    corners_trans[1] = [img_width - 1, 0]
    corners_trans[2] = [0, img_height - 1]
    corners_trans[3] = [img_width - 1, img_height - 1]

    transform = cv2.getPerspectiveTransform(corners, corners_trans)
    dst = cv2.warpPerspective(img, transform, (int(img_width), int(img_height)))
    return dst

# 对图像进行排序-将一个四边形区域的四个顶点坐标按照顺时针方向排序，以便于后续的裁剪操作
# 接受一个坐标参数，然后使用极坐标排序来将坐标按照顺时针方向排列
def order_point(coor):
    # 将coor参数转换为一个numpy数组，并将其形状改为[4,2]，表示四行两列
    arr = np.array(coor).reshape([4, 2])
    # 计算数组中所有坐标的和，并除以4，得到四边形区域的中心点坐标
    sum_ = np.sum(arr, 0)
    centroid = sum_ / arr.shape[0]
    # 计算每个坐标与中心点之间的夹角，使用numpy.arctan2函数，得到一个包含四个角度值的数组
    theta = np.arctan2(arr[:, 1] - centroid[1], arr[:, 0] - centroid[0])
    # 根据角度值对数组进行升序排序，使用numpy.argsort函数，得到一个包含四个索引值的数组
    sort_points = arr[np.argsort(theta)]
    # 将数组形状改为[4,-1]，表示四行任意列
    sort_points = sort_points.reshape([4, -1])
    # 检查第一个坐标是否在中心点的右侧，如果是，则将数组旋转一次，使得第一个坐标在中心点的左侧。这是为了保证顺时针方向
    if sort_points[0][0] > centroid[0]:
        sort_points = np.concatenate([sort_points[3:], sort_points[:3]])
    # 将数组形状改为[4,2]，表示四行两列，并将其转换为浮点类型  
    sort_points = sort_points.reshape([4, 2]).astype('float32')
    return sort_points

# 创建两个OCR流程
# 用于文本检测
ocr_detection = pipeline(Tasks.ocr_detection, model='damo/cv_resnet18_ocr-detection-line-level_damo')
# 用于文本识别
ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-general_damo')

img_path = './img/test-invoice.png'
image_full = cv2.imread(img_path)
# 调用文本检测流程
det_result = ocr_detection(image_full)
# 得到一个多边形数组，表示文本区域的位置
det_result = det_result['polygons'] 
# 处理每个多边形区域
for i in range(det_result.shape[0]):
    # 先调用order_point函数进行排序
    pts = order_point(det_result[i])
    # 再调用crop_image函数进行裁
    image_crop = crop_image(image_full, pts)
    #调用文本识别流程，得到了一个结果字典
    result = ocr_recognition(image_crop)
    # 打印出每个区域的位置和文本
    print("box: %s" % ','.join([str(e) for e in list(pts.reshape(-1))]))
    print("text: %s" % result['text'])