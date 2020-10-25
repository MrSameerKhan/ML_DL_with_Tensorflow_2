
from PIL import Image
import numpy as np
from numpy import asarray
import cv2

pilImage = Image.open("mainPY/custom/images/china1.jpg")
cvImage = cv2.imread("mainPY/custom/images/china1.jpg")
cvData = asarray(cvImage)
pilData = asarray(pilImage)

print("CV ",cvData.shape)
print("PIL", pilData.shape)

cvGray = cv2.cvtColor(cvData, cv2.COLOR_BGR2GRAY)

print("CV Gray", cvGray.shape)
cv2.imshow("CV Gray", cvGray)
cv2.waitKey(0)
cv2.destroyAllWindows()

cvSave = Image.fromarray(cvGray)
cvSave.save("mainPY/custom/images/grayChina1.jpg")