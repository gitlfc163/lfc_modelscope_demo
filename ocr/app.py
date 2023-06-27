# 读光OCR-多场景文字识别

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import numpy as np
import cv2
import math
import gradio as gr
from PIL import ImageDraw

from torchvision import transforms
from PIL import Image
import pandas as pd

# 根据给定的位置参数，对图片进行裁剪和变换，返回裁剪后的图片
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

# 对给定的坐标点进行排序，返回排序后的坐标点
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
    # 根据索引值重新排列原始数组，得到一个按照极坐标顺序排列的数组
    # 将数组形状改为[4,-1]，表示四行任意列
    sort_points = sort_points.reshape([4, -1])
    # 检查第一个坐标是否在中心点的右侧，如果是，则将数组旋转一次，使得第一个坐标在中心点的左侧
    if sort_points[0][0] > centroid[0]:
        sort_points = np.concatenate([sort_points[3:], sort_points[:3]])
    # 将数组形状改为[4,2]，表示四行两列，并将其转换为浮点类型
    sort_points = sort_points.reshape([4, 2]).astype('float32')
    return sort_points

title = "读光OCR-多场景文字识别"
description = "给定图片作为输入，选择相应场景，我们的模型会输出图片中文字行的坐标位置和识别结果。本页面提供了在线体验的服务，欢迎使用！"
examples = [
    ['./img/ocr_general.jpg',"通用场景"], #,'./img/test-invoice.png',"通用场景"
    ['./img/license_plate_detection.jpg',"车牌场景"], 
    ['./img/ocr_scene.jpg',"自然场景"], 
    ['./img/ocr_table2.jpg',"文档场景"], 
    ['./img/ocr_handwriting.jpg',"手写场景"]
    ]
#examples = ['./ocr_spotting.jpg', './license_plate_detection.jpg', './ocr_scene.jpg', './ocr_table.jpg', './ocr_handwriting.jpg']
# 加载不同场景的模型
# 通用场景模型
ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-general_damo')
# 用于文本检测
ocr_detection = pipeline(Tasks.ocr_detection, model='damo/cv_resnet18_ocr-detection-line-level_damo')
# 单词检测模型
# ocr_detection = pipeline(Tasks.ocr_detection, model='damo/cv_resnet18_ocr-detection-word-level_damo')
# 自然场景模型
ocr_recognition_scene = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-scene_damo')
# 手写场景模型
ocr_recognition_handwritten = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-handwritten_damo')
# 文档场景模型
ocr_recognition_document = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-document_damo')
# 车牌场景模型
license_plate_detection = pipeline(Tasks.license_plate_detection, model='damo/cv_resnet18_license-plate-detection_damo')
# 车牌场景模型
ocr_recognition_licenseplate = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-licenseplate_damo')
# 图像类型选择
types_dict = {"通用场景":ocr_recognition, "自然场景":ocr_recognition_scene, "手写场景":ocr_recognition_handwritten, "文档场景":ocr_recognition_document, "车牌场景":ocr_recognition_licenseplate}

# 定义返回结果
class InvoicesResult:
    # 发票类型
    invoiceType=''
    # 发票号码
    invoiceCode=''
    # 开票日期
    invoiceDate=''
    # 付款单位
    payerUnitName=''
    # 付款单位纳税人识别号
    ptaxAccounts=''
    # 付款单位客户地址
    paddress=''
    # 付款单位客户电话
    ptlephone=''
    # 付款单位开户行
    pbankName=''
    # 付款单位银行账号
    pbankAccounts=''

    # 开票内容
    invoiceContent=''
    # 开票金额（元）
    amount=0.0
    # 价税合计(大写)
    capital=''
    # 税率
    invoiceTaxType=''
    # 价税合计（元）
    invoiceTaxValue=0.0
    # 不含税额（元）
    excludingTaxValue=0.0

    # 客户名称
    customer=''
    # 纳税人识别号
    taxAccounts=''
    # 客户地址
    address=''
    # 客户电话
    tlephone=''
    # 开户行
    bankName=''
    # 银行账号
    bankAccounts=''
    # 开票人

    # __init__
    def __init__(self):
        self.invoiceType=''
        self.invoiceCode=''
        self.invoiceDate=''
        self.amount=0.0
        self.capital=''
        self.invoiceContent=''
        self.invoiceTaxType=''
        self.invoiceTaxValue=0.0
        self.excludingTaxValue=0.0
        self.payerUnitName=''
        self.ptaxAccounts=''
        self.paddress=''
        self.ptlephone=''
        self.pbankName=''
        self.pbankAccounts=''
        self.customer=''
        self.taxAccounts=''
        self.address=''
        self.tlephone=''
        self.bankName=''
        self.bankAccounts=''
        
    def set_invoice(self, invoiceType,invoiceCode,invoiceDate,amount,capital,invoiceContent,invoiceTaxType
                 ,invoiceTaxValue,excludingTaxValue,payerUnitName,ptaxAccounts,paddress,ptlephone
                 ,pbankName,pbankAccounts,customer,taxAccounts,address,tlephone
                 ,bankName,bankAccounts):
        self.invoiceType = invoiceType
        self.invoiceCode = invoiceCode
        self.invoiceDate = invoiceDate
        self.amount = amount
        self.capital = capital
        self.invoiceContent = invoiceContent

        self.invoiceTaxType = invoiceTaxType
        self.invoiceTaxValue = invoiceTaxValue
        self.excludingTaxValue = excludingTaxValue
        self.payerUnitName = payerUnitName
        self.ptaxAccounts = ptaxAccounts
        self.paddress = paddress
        self.ptlephone = ptlephone
        self.pbankName = pbankName
        self.pbankAccounts = pbankAccounts

        self.customer = customer
        self.taxAccounts = taxAccounts
        self.address = address
        self.tlephone = tlephone
        self.bankName = bankName
        self.bankAccounts = bankAccounts

# draw_boxes用于在图片上绘制文字行的边框，并将边框内的图片传递给相应的识别模型，返回识别结果
def draw_boxes(image_full, det_result):
    image_full = Image.fromarray(image_full)
    draw = ImageDraw.Draw(image_full)
    for i in range(det_result.shape[0]):
        # import pdb; pdb.set_trace()
        p0, p1, p2, p3 = order_point(det_result[i])
        draw.text((p0[0]+5, p0[1]+5), str(i+1), fill='blue', align='center')
        draw.line([*p0, *p1, *p2, *p3, *p0], fill='green', width=5)
    image_draw = np.array(image_full)
    return image_draw

# 文字检测处理
def text_detection(image_full, ocr_detection):
    det_result = ocr_detection(image_full)
    det_result = det_result['polygons']
    # sort detection result with coord
    det_result_list = det_result.tolist()
    det_result_list = sorted(det_result_list, key=lambda x: 0.01*sum(x[::2])/4+sum(x[1::2])/4)     
    return np.array(det_result_list)

# 增值税普通发票结果处理
def set_invoice(i:int,result,invoice:InvoicesResult):
    result2=result[0]
    if(i==1): # 发票类型 # 311.0,27.0,504.0,28.0,504.0,48.0,311.0,48.0
        invoice.invoiceType=result2
    elif(i==4): # 发票号码 # 546.0,42.0,656.0,43.0,656.0,62.0,546.0,62.0
        invoice.invoiceCode=result2
    elif(i==9): # 开票日期
        invoice.invoiceDate=result2
    elif(i==12):# 付款单位
        invoice.payerUnitName=result2
    elif(i==16):# 付款单位纳税人识别号
        invoice.ptaxAccounts=result2
    elif(i==21):# 付款单位地址、电话
        invoice.paddress=result2
    elif(i==16):# 付款单位纳税人识别号
        invoice.ptaxAccounts=result2
    elif(i==27):# 付款单位开户行及账号
        invoice.pbankName=result2
    elif(i==38):# 开票内容
        invoice.invoiceContent=result2
    elif(i==41):# 单价
        invoice.invoiceContent=result2
    elif(i==42):# 不含税额（元）
        invoice.excludingTaxValue=result2
    elif(i==43):# 税率
        invoice.invoiceTaxType=result2
    elif(i==44):# 税额
        invoice.invoiceTaxValue=result2
    elif(i==44):# 价税合计(大写)
        invoice.capital=result2
    elif(i==44):# 价税合计(小写)
        invoice.invoiceTaxValue=result2
    elif(i==58):# 客户单位
        invoice.customer=result2
    elif(i==61):# 客户纳税人识别号
        invoice.taxAccounts=result2
    elif(i==66):# 客户地址、电话
        invoice.address=result2
    elif(i==71):# 客户纳税人识别号
        invoice.ptaxAccounts=result2
    elif(i==70):# 客户开户行及账号
        invoice.pbankName=result2 
    return invoice

# 文本识别处理
def text_recognition(det_result, image_full, ocr_recognition):
    output = []
    invoice=InvoicesResult()
    for i in range(det_result.shape[0]):
        pts = order_point(det_result[i])
        image_crop = crop_image(image_full, pts)
        result = ocr_recognition(image_crop)
        invoice=set_invoice(i+1,result['text'],invoice)
        output.append([str(i+1), result['text'], ','.join([str(e) for e in list(pts.reshape(-1))])])
    result = pd.DataFrame(output, columns=['检测框序号', '行识别结果', '检测框坐标'])
    print(invoice)
    return result

# 一键识别处理函数
def text_ocr(image_full, types='通用场景'):
    if types == '车牌场景':
        det_result = text_detection(image_full, license_plate_detection)
        ocr_result = text_recognition(det_result, image_full, ocr_recognition_licenseplate)
        image_draw = draw_boxes(image_full, det_result)
    else:
        det_result = text_detection(image_full, ocr_detection)
        ocr_result = text_recognition(det_result, image_full, types_dict[types])        
        image_draw = draw_boxes(image_full, det_result)
        # print(ocr_result.values)
    return image_draw, ocr_result 


with gr.Blocks() as demo:
    gr.Markdown(description)
    with gr.Row():
        select_types = gr.Radio(label="图像类型选择", choices=["通用场景", "自然场景", "手写场景", "文档场景", "车牌场景"], value="通用场景")
    with gr.Row():
        img_input = gr.Image(label='输入图像', elem_id="fixed_size_img")
        img_output = gr.Image(label='图像可视化效果', elem_id="fixed_size_img")
    with gr.Row():
        btn_submit = gr.Button(value="一键识别")        
    with gr.Row():
        text_output = gr.components.Dataframe(label='识别结果', headers=['检测框序号', '行识别结果', '检测框坐标'], wrap=True)
    with gr.Row():
        examples = gr.Examples(examples=examples, inputs=[img_input, select_types], label='点击示例图片体验OCR效果' , outputs=[img_output, text_output], fn=text_ocr, cache_examples=True)

    btn_submit.click(fn=text_ocr, inputs=[img_input, select_types], outputs=[img_output, text_output])

    demo.launch(server_name="172.17.70.46",server_port=8097)