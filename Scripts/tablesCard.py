
import cv2
import numpy as np
import os
import math
from scipy import ndimage

sourceImage = cv2.imread("/home/sameer/Music/ML_DL_with_Tensorflow_2/Scripts/CARDS/26161761_4_2.jpg", 0)
image = cv2.imread("/home/sameer/Music/ML_DL_with_Tensorflow_2/Scripts/CARDS/26161761_4_2.jpg", 0)

thresh, img_bin = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
img_bin = 255 - img_bin

kernel_length_ver = np.array(image).shape[1] // 1
kernel_length_hor = np.array(image).shape[1]//500

vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length_ver))

horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length_hor, 1))

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

img_temp1 = cv2.erode(img_bin, vertical_kernel, iterations=3)
verticle_lines_img = cv2.dilate(img_temp1, vertical_kernel, iterations=3)

img_temp2 = cv2.erode(img_bin, horizontal_kernel, iterations=3)
horizontal_lines_img = cv2.dilate(img_temp2, horizontal_kernel, iterations=3)

alpha = 0.5
beta = 1.0 - alpha

img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
(thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

img_edges = cv2.Canny(img_final_bin, 100, 100, apertureSize=3)
lines = cv2.HoughLinesP(img_edges, 1 , math.pi/ 180.0, 100, minLineLength=100, maxLineGap=10)
angles = []
i = 0

try:
    
    for x1, y1, x2, y2 in lines[0]:
        cv2.line(img_final_bin, (x1, y1), (x2, y2), (255,0,0),3)

        angle = math.degrees(math.atan2(y2-y1, x2-x1))
        angles.append(angle)
        median_angle = np.medium(angles)
        rotatedImage = ndimage.rotate(sourceImage, median_angle)
        

except:
    pass



cv2.imshow("rotatedImage", rotatedImage)
cv2.waitKey(0)
