import paddle,PyInstaller
import os,re,sys
import openpyxl,PIL
from tkinter import *
from tkinter import filedialog
from paddleocr import PaddleOCR, draw_ocr


# 遍历文件夹及其子文件夹中的文件，并存储在一个列表中
# 输入文件夹路径、空文件列表[]
# 返回 文件列表Filelist,包含文件名（完整路径）
def get_filelist(dir, Filelist):
    newDir=dir
    if os.path.isfile(dir):
        Filelist.append(dir)
        # # 若只是要返回文件文，使用这个
        # Filelist.append(os.path.basename(dir))
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
        #如果需要忽略某些文件夹，使用以下代码
        #if s == "xxx":
        #continue
            newDir=os.path.join(dir, s)
            get_filelist(newDir, Filelist)
    return Filelist

def get_files(Filelist,dat):
    paths = []
    names=[]
    for file in Filelist:
        if file.split('.')[-1] in dat:
                paths.append(file)
                names.append(file.split('\\')[-1])
    return paths,names

# Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
# 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
def fp_ocr(img_path,name):
    dic = {}
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory
    result = ocr.ocr(img_path, cls=True)
    txts = [line[1][0] for line in result]
    text=' '.join(txts)
    print(text)
    gongsi = re.findall(r'[一-龟]+公司|个人', text)
    gongsi = [g for g in gongsi if '银行' not in g]
    dic['文件名'] = name
    dic['发票代码'] = re.findall(r'\d{12}', text)[0]
    dic['发票号码'] = re.findall(r'\d{8}', text)[1]
    dic['开票日期'] = re.findall(r'[\d ]+年[\d ]+月[\d ]+日', text)[0]
    dic['校验码'] = re.findall(r'[\d ]{20,30}', text)[0]
    try:
        dic['税率'] = re.findall(r'免税|不征税|1{0,1}[1369]%', text)[-1]
    except:
        dic['税率'] = ''
    dic['价税合计(小写)'] = re.findall(r'[¥|￥]{0,1}\d+\.\d+', text)[-1]
    dic['购买方名称'] = gongsi[0]
    dic['销售方名称'] = gongsi[-1]
    '''
    da = re.findall(r'(?<=公司 )[\w\W]{0,}?纳税人识别号', text)
    if len(da)>1:
        dic['备注'] = re.sub(r'[ 销备售纳税人识别]{0,}', '', da[-1])
    else:
        dic['备注'] = ''
    '''
    #规范格式化处理
    dic['开票日期'] = re.sub(r'年|月|日| ', '', dic['开票日期'])
    dic['校验码'] = dic['校验码'].replace(' ', '')
    print('已识别完成：{}'.format(name))
    return dic

if __name__=="__main__":
    root = Tk()
    root.withdraw()
    ####rootDir = str(Path(sys.argv[0]).parent)
    rootDir =filedialog.askdirectory()#指定存储电子发票的文件夹
    filelist = get_filelist(rootDir, [])#遍历文件夹里的全部文件
    paths, names = get_files(filelist, ['jpg','png'])#判断PDF文件，返回PDF文件的路径和文件名
    data=[]
    for n in range(len(paths)):
        dic=fp_ocr(paths[n], names[n])
        data+=[dic]
    #保存到Excel
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(list(dic.keys()))
    for d in data:
        ws.append(list(d.values()))
    wb.save(rootDir+"\\已识别发票信息.xlsx")
    print('一共识别完{}张发票jpg或png图片'.format(n))
    # print("谢谢使用本脚本：By所长_WCEO QQ121841879", "\n")
    ppp = input("程序已运行完成，输入任意内容结束程序：")