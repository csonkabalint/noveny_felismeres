import matplotlib.pyplot as plt
import cv2
import numpy as np


def show_end():
    plt.show()
    exit(0)

img = cv2.imread("imageProcessTestData/quercus_acutissima_04.jpg")
#img = cv2.imread("imageProcessTestData/quercus_acutissima_03.jpg")
#img = cv2.imread("imageProcessTestData/quercus_lobata_01.jpg")
#img = cv2.imread("imageProcessTestData/quercus_agrifolia_01.jpg")
#img = cv2.imread("imageProcessTestData/quercus_robus_01.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
#img = cv2.medianBlur(img, 5)
B, G, R = cv2.split(img)
print(type(B))



# VARI
vari = (G - R) / (G + R - B + 0.00001)
#plt.figure()
#plt.imshow(vari)


# J
j = (2*G - R - B) - (2*R - G - B)
plt.figure()
plt.imshow(j)



th, otsu_bin = cv2.threshold(j, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.figure()
plt.imshow(otsu_bin)
otsu_bin = cv2.bitwise_not(otsu_bin)

plt.figure()
plt.imshow(otsu_bin)

img_osu_mask = cv2.bitwise_and(img, img, mask=otsu_bin)

mask_i = img_osu_mask.sum(axis=2)
mask_i = mask_i / 3
print(mask_i.shape)

plt.figure()
plt.imshow(mask_i)


mask_i8 = mask_i.astype(np.uint8)
mask_i8 = cv2.medianBlur(mask_i8, 3)
mask_i8 = cv2.medianBlur(mask_i8, 3)
mask_i8 = cv2.medianBlur(mask_i8, 3)
i_edges = cv2.Canny(mask_i8, 10, 220)

plt.figure()
plt.imshow(i_edges)

show_end()

# NDVI
ndvi = (G - R) / (G + R)
#plt.figure()
#plt.imshow(ndvi)




img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lowerBound, upperBound = (36, 0, 0), (86, 255, 255)
kernelOpen = np.ones((5, 5))
kernelClose = np.ones((20, 20))
H, S, V = cv2.split(img_hsv)
mask = cv2.inRange(img_hsv, lowerBound, upperBound)
maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)


img_edges = cv2.Canny(V, 10, 220)
kernel0 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
img_edges = cv2.dilate(img_edges, kernel0)

plt.figure()
plt.imshow(img_edges)

edge_oshu = cv2.bitwise_and(img_edges, img_edges, mask=otsu_bin)
plt.figure()
plt.imshow(edge_oshu)

masked_V = cv2.bitwise_and(V, V, mask=maskClose)
#water_shed = cv2.watershed(masked_V)

masked_V = cv2.medianBlur(masked_V, 3)
edges = cv2.Canny(masked_V, 150, 200)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9, 9))
dilated = cv2.dilate(edges, kernel)
cnts, b = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(type(cnts))
print(b)

kernel2 = np.ones((1, 20), np.uint8)  # note this is a horizontal kernel
d_im = cv2.dilate(dilated, kernel2, iterations=1)
e_im = cv2.erode(d_im, kernel2, iterations=1)

plt.figure()
plt.imshow(e_im)

plt.figure()
plt.imshow(maskClose)

image_2 = (otsu_bin + mask) / 2
image_2 = cv2.morphologyEx(image_2, cv2.MORPH_CLOSE, kernelClose)
#plt.figure()
#plt.imshow(image_2)

plt.show()


