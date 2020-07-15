import cv2
import numpy as np
img = cv2.imread('girl2.png',cv2.IMREAD_UNCHANGED)
arry = np.array(img).tolist()
#cv2.boxFilter(src,depth,ksize,normalize)

r = cv2.boxFilter(img, -1 , (7,7) , normalize = 1)
d = cv2.boxFilter(img, -1 , (3,3) , normalize = 0)
cv2.namedWindow('img',cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('r',cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('d',cv2.WINDOW_AUTOSIZE)
cv2.imshow('img',img)
cv2.imshow('r',r)
cv2.imshow('d',d)
cv2.waitKey(0)
cv2.destroyAllWindows()