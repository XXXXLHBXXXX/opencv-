import cv2

def sort_contours(cnts, method = "left-2-right"):
    reverse = False
    i = 0

    if method == "right-2-left" or method == "bottom-2-top":
        reverse = True
    if method == "top-2-bottom" or method == "bottom-2-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    #使用一个最小的举行，将找到的形状包起来x,y,h,w
    (cnts,boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                       key=lambda b:b[1][i],reverse=reverse))

    return cnts,boundingBoxes

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h,w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r),height)
    else:
        r = width / float(w)
        dim = (width,int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized
