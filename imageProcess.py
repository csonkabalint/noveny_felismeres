import matplotlib.pyplot as plt
import cv2
import numpy as np
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters


def calc_ExGR(cB, cG, cR):
    ExG = (2*cG)-cR-cB
    print(type(ExG))
    ExR = (1.4*cR)-cB
    ExGR = ExG - ExR
    return ExGR

def show_end():
    plt.show()
    exit(0)


def calculate_i_channel(img_a):
    img_a = img_a.sum(axis=2)
    img_a = img_a / 3
    return img_a


def show_image(img_b, title):
    fig = plt.figure()
    fig.canvas.set_window_title(title)
    plt.imshow(img_b)


def calculate_j(img_c):
    B, G, R = cv2.split(img_c)
    j = (2*G - R - B) - (2*R - G - B)
    return j

def blur_times(img_d, times):
    for i in range(times):
        img_d = cv2.medianBlur(img_d, 3)
    return img_d


def structured_edge_bgr(img_e):

    #img_e_orig = img_e.copy()
    image_e = cv2.cvtColor(img_e, cv2.COLOR_BGR2RGB)
    image_e = image_e.astype(np.float32) / 255.0
    """
    gray_e = cv2.cvtColor(img_e_orig, cv2.COLOR_BGR2GRAY)
    blurred_e = cv2.GaussianBlur(gray_e, (5, 5), 0)
    """
    edge_detector_e = cv2.ximgproc.createStructuredEdgeDetection('StructuredEdgeModel/model.yml')
    edges_e = edge_detector_e.detectEdges(image_e)
    return edges_e

def structured_edge(img_f):
    img_f = img_f.astype(np.float32) / 255.0
    edge_detector_e = cv2.ximgproc.createStructuredEdgeDetection('StructuredEdgeModel/model.yml')
    edges_e = edge_detector_e.detectEdges(img_f)
    return edges_e

def structured_edge_one(img_e):
    # img_e_orig = img_e.copy()
    image_e = img_e.astype(np.float32) / 255.0
    """
    gray_e = cv2.cvtColor(img_e_orig, cv2.COLOR_BGR2GRAY)
    blurred_e = cv2.GaussianBlur(gray_e, (5, 5), 0)
    """
    edge_detector_e = cv2.ximgproc.createStructuredEdgeDetection('StructuredEdgeModel/model.yml')
    edges_e = edge_detector_e.detectEdges(image_e)
    return edges_e


kernelOpen = np.ones((10, 10))
kernelClose = np.ones((20, 20))


#img = cv2.imread("imageProcessTestData/quercus_acutissima_01.jpg")
##img = cv2.imread("imageProcessTestData/quercus_acutissima_02.jpg")
##img = cv2.imread("imageProcessTestData/quercus_acutissima_04.jpg")
##img = cv2.imread("imageProcessTestData/quercus_acutissima_03.jpg")
#img = cv2.imread("imageProcessTestData/quercus_acutissima_05.jpg")
#img = cv2.imread("imageProcessTestData/quercus_lobata_01.jpg")
#img = cv2.imread("imageProcessTestData/quercus_agrifolia_01.jpg")
#img = cv2.imread("imageProcessTestData/quercus_robus_01.jpg")
img = cv2.imread("imageProcessTestData/level1.jpg")
#img = cv2.imread("imageProcessTestData/level2.jpg")
#img = cv2.imread("imageProcessTestData/level3.jpg")

hh, ww = img.shape[:2]

img = cv2.resize(img, (int(ww / 4), int(hh / 4)), interpolation=cv2.INTER_AREA)


img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
#img = cv2.medianBlur(img, 5)
B, G, R = cv2.split(img)
print(type(B))

"""
img = cv2.GaussianBlur(img, (5, 5), 0)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
b = clahe.apply(img[:, :, 0])
g = clahe.apply(img[:, :, 1])
r = clahe.apply(img[:, :, 2])
equalized = np.dstack((b, g, r))

eq_rgb = cv2.cvtColor(equalized, cv2.COLOR_BGR2RGB)
#show_image(eq_rgb, "eq_rgb")


img = equalized
img = cv2.blur(img, (5, 5))
"""
#img = equalized
#img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# VARI
#vari = (G - R) / (G + R - B + 0.00001)
#show_image(vari, "vari")
#vari = vari.astype(np.uint8)
#print(vari.dtype)
#show_image(vari, "vari")
#plt.figure()
#plt.imshow(vari)


# J
#img_j = calculate_j(img)
#show_image(img_j)

img_exgr = calc_ExGR(B, G, R)
print(type(img_exgr))
show_image(img_exgr, "img_exgr")
img_exgr = img_exgr.astype(np.uint8)


th_e, exgr_bin = cv2.threshold(img_exgr, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
show_image(exgr_bin, "exgr_bin")

exgr_bin = cv2.morphologyEx(exgr_bin, cv2.MORPH_OPEN, kernelOpen)
show_image(exgr_bin, "exgr_bin")
exgr_bin = cv2.morphologyEx(exgr_bin, cv2.MORPH_CLOSE, kernelClose)
show_image(exgr_bin, "exgr_bin")
exgr_bin_not = cv2.bitwise_not(exgr_bin)
"""

img_exgr_mask = cv2.bitwise_and(img, img, mask=exgr_bin)
show_image(img_exgr_mask, "img_exgr_mask")
"""

"""
th, otsu_bin = cv2.threshold(img_j, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print(th)
show_image(otsu_bin,"otsu_bin")
otsu_bin = cv2.morphologyEx(otsu_bin, cv2.MORPH_OPEN, kernelOpen)
show_image(otsu_bin,"otsu_bin")
otsu_bin = cv2.morphologyEx(otsu_bin, cv2.MORPH_CLOSE, kernelClose)
show_image(otsu_bin,"otsu_bin")
otsu_bin = cv2.bitwise_not(otsu_bin)
"""

"""
th, otsu_bin = cv2.threshold(vari, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print(th)
show_image(otsu_bin,"otsu_bin")
otsu_bin = cv2.morphologyEx(otsu_bin, cv2.MORPH_OPEN, kernelOpen)
show_image(otsu_bin,"otsu_bin")
otsu_bin = cv2.morphologyEx(otsu_bin, cv2.MORPH_CLOSE, kernelClose)
show_image(otsu_bin,"otsu_bin")
otsu_bin = cv2.bitwise_not(otsu_bin)

show_end()
"""
#show_image(otsu_bin, "A")


#img = cv2.bitwise_and(img, img, mask=otsu_bin)
#img_osu_mask = cv2.bitwise_and(img, img, mask=otsu_bin)
#show_image(img_osu_mask, "img_osu_mask")
"""
orig_image = img_osu_mask.copy()
image = cv2.cvtColor(img_osu_mask, cv2.COLOR_BGR2RGB)
image = image.astype(np.float32) / 255.0
gray = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

edge_detector = cv2.ximgproc.createStructuredEdgeDetection('StructuredEdgeModel/model.yml')
edges = edge_detector.detectEdges(image)
"""
"""
edges_i = calculate_i_channel(img)
edges_i = edges_i.astype(int)
print(type(edges_i))
print(edges_i.shape)
print(edges_i)
print(type(img))
print(img.shape)
print(img)
"""
#edges_i = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
"""
frame = img

backSub = cv2.createBackgroundSubtractorMOG2()
fgMask = backSub.apply(frame)

cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
#cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

show_image(frame, "Frame")
show_image(fgMask, "FG Mask")
"""


#edges_i = structured_edge(edges_i)
#show_image(edges_i, "edges_i")
edges_canny = cv2.Canny(img,60,120)
show_image(edges_canny, "edges_canny")


"""
edges_canny = cv2.morphologyEx(edges_canny, cv2.MORPH_CLOSE, kernelClose)
show_image(edges_canny, "edges_canny")
edges_canny = cv2.morphologyEx(edges_canny, cv2.MORPH_OPEN, kernelOpen)
show_image(edges_canny, "edges_canny")
"""

edges_orig = structured_edge_bgr(img)
show_image(edges_orig, "edges_orig")

"""
edges_orig = cv2.morphologyEx(edges_orig, cv2.MORPH_CLOSE, kernelClose)
show_image(edges_orig, "edges_orig")
edges_orig = cv2.morphologyEx(edges_orig, cv2.MORPH_OPEN, kernelOpen)
show_image(edges_orig, "edges_orig")
"""

#structured_canny = cv2.Canny(img, 60, 120)
#show_image(structured_canny, "structured_canny")

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv_structured = structured_edge(img_hsv)
show_image(hsv_structured, "hsv_structured")

H, S, V = cv2.split(img_hsv)
VVV = np.dstack([V, V, V])
VVV_structured = structured_edge(VVV)
show_image(VVV_structured, "VVV_structured")

exgr3 = np.dstack([img_exgr, img_exgr, img_exgr])
exgr_structured = structured_edge(exgr3)
show_image(exgr_structured, "exgr_structured")

orig_plus_exgr3 = edges_orig + exgr_structured
show_image(orig_plus_exgr3, "orig_plus_exgr3")
#orig_and_canny = cv2.bitwise_and(edges_orig, edges_orig, mask=edges_canny)
#show_image(orig_and_canny, "orig_and_canny")



#edges_orig = cv2.bitwise_and(edges_orig, edges_orig, mask=otsu_bin)


#uh = cv2.morphologyEx(edges_orig, cv2.MORPH_OPEN, kernelOpen)
#uh = cv2.morphologyEx(uh, cv2.MORPH_OPEN, kernelOpen)
#uh = cv2.morphologyEx(uh, cv2.MORPH_OPEN, kernelOpen)
#uh = cv2.morphologyEx(edges_orig, cv2.MORPH_CLOSE, kernelClose)
#uh = cv2.morphologyEx(uh, cv2.MORPH_CLOSE, kernelClose)
#uh = cv2.morphologyEx(uh, cv2.MORPH_CLOSE, kernelClose)

#element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5), (-1, -1))
#uh = cv2.erode(edges_orig, element)

#uh = edges_orig.copy()
#uh = uh*uh*uh*uh*uh
#uh = (edges_orig > 0.2 * max(map(max, edges_orig))) * edges_orig

#edges_orig = edges_orig + 1
#edges_orig = edges_orig * edges_orig * edges_orig * edges_orig
show_image(edges_orig, "edges_orig")
uh = orig_plus_exgr3.copy()
uh = (uh > 0.2) * uh
#uh = edges_canny.copy()
show_image(uh, "uh")


#edges_orig_blurred = blur_times(img, 100)
#edges_orig_blurred = structured_edge(img)
#show_image(edges_orig_blurred, "edges_orig_blurred")

#edges = structured_edge(img_osu_mask)
#show_image(edges, "edges")

#edge_i_and_orig = cv2.bitwise_and(edges_i, edges_orig)

#show_image(edge_i_and_orig, "edge_i_and_orig")

#edge_i_and_orig_and_or = edge_i_and_orig = cv2.bitwise_or(edge_i_and_orig, uh)
#show_image(edge_i_and_orig_and_or, "edge_i_and_orig_and_or")

#mask_i = calculate_i_channel(img_osu_mask)

#show_image(mask_i, "mask_i")


#mask_i8 = mask_i.astype(np.uint8)

#i_edges = cv2.Canny(mask_i8, 10, 220)

#show_image(i_edges, "i_edges")
""""""
hh, ww = edges_orig.shape[:2]
print(hh)
print(ww)
# resize down, then back up

print(img.shape)
"""
eo1 = edges_orig
eo1 = eo1 * 255
eo1 = eo1.astype(np.uint8)
eo1 = (eo1 > 100) * eo1
#edges_orig3 = np.zeros((hh, ww), dtype=np.int32)


print(edges_orig3.shape)

show_image(edges_orig3, "edges_orig3")
"""

#uh = edges_orig

to_resize = uh.copy()
to_resize = cv2.bitwise_and(to_resize, to_resize, mask=exgr_bin_not)

#to_resize = exgr_bin.copy()

rsize = 32
h = rsize
w = (ww/hh)*rsize

#h, w = (hh/32, ww/32)
result_0 = cv2.resize(to_resize, (int(w), int(h)), interpolation=cv2.INTER_AREA)
result_0 = cv2.blur(result_0, (2, 2))
show_image(result_0, "pixelated_0")
result_1 = cv2.resize(result_0, (ww, hh), interpolation=cv2.INTER_AREA)

#show_image(result_1, "pixelated_1")



neighborhood_size = 5
threshold = 0.005

data = result_0

data_max = ndimage.maximum_filter(data, neighborhood_size)
maxima = (data == data_max)

data_min = ndimage.minimum_filter(data, neighborhood_size)
minima = (data == data_min)

diff = ((data_max - data_min) > threshold)

maxima[diff == 0] = 0
minima[diff == 0] = 0


labeled, num_objects = ndimage.label(minima)
slices_min = ndimage.find_objects(labeled)
x_min, y_min = [], []
for dy, dx in slices_min:
    x_center = (dx.start + dx.stop - 1)/2
    x_min.append(x_center)
    y_center = (dy.start + dy.stop - 1)/2
    y_min.append(y_center)


labeled, num_objects = ndimage.label(maxima)
slices = ndimage.find_objects(labeled)
x, y = [], []
for dy, dx in slices:
    x_center = (dx.start + dx.stop - 1)/2
    x.append(x_center)
    y_center = (dy.start + dy.stop - 1)/2
    y.append(y_center)

print(x_min)


plt.imshow(data)
show_image(data, "data")

plt.autoscale(False)
plt.plot(x_min, y_min, 'ro')

print(hh/h)


for i in range(len(x_min)):
    x_min[i] = (int(hh/h))*x_min[i]
    y_min[i] = (int(ww/w))*y_min[i]

plt.figure()
plt.imshow(edges_orig)
plt.autoscale(False)
plt.plot(x_min, y_min, 'ro')

print(x_min)
print(y_min)

marker = np.zeros(edges_orig.shape, dtype=np.uint8)


for i in range(len(x_min)):
    #marker[int(x_min[i]), int(y_min[i])] = 1
    marker[int(y_min[i]), int(x_min[i])] = 1

show_image(marker, "marker")

ret, markers = cv2.connectedComponents(marker)

#markers = markers+1


#wts = cv2.watershed(img, markers)

#print(edges_orig3.dtype)

e = edges_orig.copy()
e = e * 255
e = e.astype(np.uint8)
show_image(e,"e")
#edges_orig3 = np.dstack([e, np.zeros(e.shape[:2]), np.zeros(e.shape[:2])])
edges_orig3 = np.dstack([e, e, e])
show_image(edges_orig3, "edges_orig3")
edges_orig3 = edges_orig3
wts_edge = cv2.watershed(edges_orig3, markers)


#wts_hsv = cv2.watershed(img_hsv, markers)

#img_gau = cv2.GaussianBlur(img, (5, 5), 0)
#show_image(img_gau, "img_gau")
#wts_blur = cv2.watershed(img_gau, markers)

#show_image(wts, "wts")
#show_image(wts_blur, "wts_blur")
#show_image(wts_hsv, "wts_hsv")
show_image(wts_edge, "wts_edge")
show_end()


#wts = wts.astype(np.uint8)

#edges_str_fin = cv2.Canny(wts, 10, 220)
#show_image(edges_str_fin, "edges_str_fin")

#print(edges_str_fin.size)
print(edges_orig.size)

#edges_str_fin_orig = cv2.bitwise_and(edges_str_fin, edges_orig)
#show_image(edges_str_fin_orig, "edges_str_fin_orig")

















# NDVI
ndvi = (G - R) / (G + R)
#plt.figure()
#plt.imshow(ndvi)




img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lowerBound, upperBound = (36, 0, 0), (86, 255, 255)

H, S, V = cv2.split(img_hsv)
mask = cv2.inRange(img_hsv, lowerBound, upperBound)
maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)


img_edges = cv2.Canny(V, 10, 220)
kernel0 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
img_edges = cv2.dilate(img_edges, kernel0)

plt.figure()
plt.imshow(img_edges)




#edge_oshu = cv2.bitwise_and(img_edges, img_edges, mask=otsu_bin)
#plt.figure()
#plt.imshow(edge_oshu)

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

#image_2 = (otsu_bin + mask) / 2
#image_2 = cv2.morphologyEx(image_2, cv2.MORPH_CLOSE, kernelClose)
#plt.figure()
#plt.imshow(image_2)

plt.show()


