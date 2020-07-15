#导入包
from imutils import contours
import numpy as np
import cv2
import argparse
import myutils
#设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-t", "--template", required=True, help="path to template OCR-A image")
args = vars(ap.parse_args())

#指定信用卡类型
FIRST_NUMBER = {
    "3":"American Express",
    "4":"Visa",
    "5":"MasterCard",
    "6":"Discover Card"
}

#绘图的展示
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#读取一个模板图像
img = cv2.imread(args["template"])
cv_show('img',img)
#转灰度图
ref = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv_show('ref',ref)
#转二值图
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
cv_show('ref',ref)

#计算轮廓
#cv2.findContours()函数接受二值图
#cv2.RETR_EXTERNAL只检测外轮廓
#cv2.CHAIN_APPROX_SIMPLE只保留终点坐标
#返回list中的每一个元素都是一个轮廓

refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img,refCnts,-1,(0,255,255),3)
cv_show('img',img)
print(np.array(refCnts).shape)
#排序，从左到右，从上到下
refCnts = myutils.sort_contours(refCnts, method="left-2-right")[0]
digits = {}

#遍历每一个轮廓
for(i, c) in enumerate(refCnts):
    #计算外接矩形并resize为合适的大小
    (x, y, w, h) = cv2.boundingRect(c)
    roi = ref[y:y+h, x:x+w]
    roi = cv2.resize(roi,(50,90))

    #每一个数字对应每一个模板
    digits[i] = roi

#初始化卷积核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5 ))

#读取输入的图像，预处理
image = cv2.imread(args["image"])
cv_show('image',image)
image = myutils.resize(image, width=300)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv_show('gray',gray)

#礼帽操作,突出明亮的区域
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
cv_show('tophat',tophat)

gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)

gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX),np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")

print(np.array(gradX).shape)
cv_show('gray', gray)

#执行闭操作，将数字连在一起
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
cv_show('grayX',gradX)
#使用大津法自动的寻找阈值，适合的双峰，把阈值设置为0
thresh = cv2.threshold(gradX, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv_show('thresh',thresh)

#再执行闭操作
thresh = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,sqKernel)
cv_show('thresh',thresh)

#计算轮廓

threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = threshCnts
cur_img = image.copy()
cv2.drawContours(cur_img, cnts, -1, (0,255,0),3)
cv_show('img',cur_img)
locs = []

#遍历轮廓
for(i, c) in enumerate(cnts):
    #计算矩形
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)

    #选择适合的区域，根据实际任务
    if ar > 2.5 and ar < 4.0:

        if(w > 40 and w < 55) and (h > 10 and h < 20):
            #符号留下
            locs.append((x, y, w, h))

#将符合的轮廓从左到右排序
locs = sorted(locs, key=lambda x:x[0])
output = []

#遍历每一个轮廓中的数字
for(i, (gX, gY, gW, gH)) in enumerate(locs):
    groupOutput = []
    #根据坐标提取每一个组
    group = gray[gY - 10:gY + gH + 10, gX - 10:gX + gW + 10]
    cv_show('group',group)
    #预处理
    group = cv2.threshold(group, 0, 255,
            cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
    cv_show('group',group)
    #计算每一组的轮廓
    digitCnts, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = contours.sort_contours(digitCnts,method="left-2-right")[0]

    #计算每一组中的每一个数值
    for c in digitCnts:
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y:y + h,x:x + w]
        roi = cv2.resize(roi,(50,90))
        cv_show('roi',roi)

        #计算匹配得分
        scores = []

        #在模板中计算每一个得分
        for(digit, digitROI) in digits.items():
            #模板匹配
            results = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(results)
            scores.append(score)

        #得到最合适的数字
        groupOutput.append(str(np.argmax(scores)))

    #画出
    cv2.rectangle(image, (gX - 10,gY - 10),(gX + gW + 10, gY + gH + 10),(255,0,255),1)
    cv2.putText(image,"".join(groupOutput),(gX, gY - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.65,(0,0,255),2)

    #得出结果
    output.extend(groupOutput)

#打印结果
print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
print("Credit Card #:{}".format("".join(output)))
cv2.imshow('image',image)
cv2.waitKey(0)
