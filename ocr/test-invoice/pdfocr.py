# 电子发票识别---python利用正则表达式匹配发票内容

import os
import pdfplumber as pb
import pandas as pd
import re
import openpyxl
from pathlib import Path
import sys
import shutil
from time import sleep
from tkinter import *
from tkinter import filedialog

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

def get_dic(page,name,m):
    dic = {}
    text = page.extract_text()
    gongsi = re.findall(r'[一-龟]+公司|个人', text)
    gongsi = [g for g in gongsi if '银行' not in g]
    dic['文件名'] = name.replace(".pdf", "_") + str(m + 1)
    dic['发票代码'] = re.findall(r'\d{12}', text)[0]
    dic['发票号码'] = re.findall(r'\d{8}', text)[1]
    da = re.findall(r'[\d ]+年[\d ]+月[\d ]+日', text)[0]
    dic['开票日期'] = re.sub(r'年|月|日| ', '', da)
    da = re.findall(r'[\d ]{20,30}', text)[0]
    dic['校验码'] = da.replace(' ', '')
    try:
        dic['税率'] = re.findall(r'免税|不征税|1{0,1}[1369]%', text)[-1]
    except:
        dic['税率'] = ''
    dic['价税合计(小写)'] = re.findall(r'[¥|￥]{0,1}\d+\.\d+', text)[-1]
    dic['购买方名称'] = gongsi[0]
    dic['销售方名称'] = gongsi[-1]
    da = re.findall(r'(?<=公司 )[\w\W]{0,}?纳税人识别号', text)
    if len(da)>1:
        dic['备注'] = re.sub(r'\n销 备 |\n纳税人识别号', '', da[-1])
    else:
        dic['备注'] = ''
    return dic

if __name__=="__main__":
    root = Tk()
    root.withdraw()
    ####rootDir = str(Path(sys.argv[0]).parent)
    rootDir =filedialog.askdirectory()#指定存储电子发票的文件夹
    filelist = get_filelist(rootDir, [])#遍历文件夹里的全部文件
    paths, names = get_files(filelist, ['pdf'])#判断PDF文件，返回PDF文件的路径和文件名
    data=[]
    dic = {}
    for n in range(len(paths)):
        pdf = pb.open(paths[n])#打开PDF文件
        m = 0
        for page in pdf.pages:#遍历PDF文件的每一页
            dic = {}
            if '开票人' in page.extract_text():#通过判断是否包含“开票人”，判断是否为电子发票（有的发票跟着行程或税务清单需排除掉）
                dic=get_dic(page,names[n],m)#获取电子发票的关键信息
                data += [dic]
            m+=1
        pdf.close()
    #保存到Excel
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(list(dic.keys()))
    for d in data:
        ws.append(list(d.values()))
    wb.save(rootDir+"\\已识别发票信息.xlsx")