import cv2
import matplotlib.pyplot as plt

img = cv2.imread("imageProcessTestData/quercus_acutissima_01.jpg")


img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img_grey, cv2.COLOR_GRAY2RGB)
plt.imshow(img_rgb)
plt.show()