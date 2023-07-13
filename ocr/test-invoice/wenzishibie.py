import csv
from tkinter import *
from tkinter import filedialog
from paddleocr import PaddleOCR, draw_ocr

if __name__=="__main__":
    root = Tk()
    root.withdraw()
    rootDir = filedialog.askopenfilenames()
    with open('/'.join(rootDir[0].split('/')[0:-1] + ['识别文字.csv']), 'a', encoding='utf-8') as f:
        for path in rootDir:
            name = path.split('/')[-1].split('.')[0]
            ocr = PaddleOCR(use_angle_cls=True,lang="ch")  # need to run only once to download and load model into memory
            result = ocr.ocr(path, cls=True)
            txts = [name]+[str(line[1][0]) for line in result]
            csv_write = csv.writer(f)#csv_write = csv.writer(f)
            csv_write.writerow(txts)#  3.构建列表头
            print('\n*******已识别一张图片*******\n')
    print("\n谢谢使用本脚本：\n")
    ppp = input("程序已运行完成，输入任意内容结束程序：")