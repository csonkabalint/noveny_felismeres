import matplotlib.pyplot as plt
import cv2
import numpy as np
import scipy.ndimage as ndimage
from PIL import Image


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


def locate_local_minimums(image, neighborhood_size, threshold):
    data_min = ndimage.minimum_filter(image, neighborhood_size)
    data_max = ndimage.maximum_filter(data, neighborhood_size)
    minima = (image == data_min)
    diff = ((data_max - data_min) > threshold)
    minima[diff == 0] = 0
    labeled = ndimage.label(minima)[0]
    slices_min = ndimage.find_objects(labeled)
    x_min, y_min = [], []
    for dy, dx in slices_min:
        x_center = (dx.start + dx.stop - 1) / 2
        x_min.append(x_center)
        y_center = (dy.start + dy.stop - 1) / 2
        y_min.append(y_center)
    return x_min, y_min


kernelOpen = np.ones((10, 10))
kernelClose = np.ones((20, 20))


#img = cv2.imread("imageProcessTestData/quercus_acutissima_01.jpg")
#img = cv2.imread("imageProcessTestData/quercus_acutissima_02.jpg")
#img = cv2.imread("imageProcessTestData/quercus_acutissima_04.jpg")
#img = cv2.imread("imageProcessTestData/quercus_acutissima_03.jpg")
#img = cv2.imread("imageProcessTestData/quercus_acutissima_05.jpg")
#img = cv2.imread("imageProcessTestData/quercus_lobata_01.jpg")
#img = cv2.imread("imageProcessTestData/quercus_agrifolia_01.jpg")
#img = cv2.imread("imageProcessTestData/quercus_robus_01.jpg")
#img = cv2.imread("imageProcessTestData/level1.jpg")
#img = cv2.imread("imageProcessTestData/level2.jpg")
#img = cv2.imread("imageProcessTestData/level3.jpg")
img = cv2.imread("imageProcessTestData/level4.jpg")
img = cv2.imread("imageProcessTestData/level7.jpg")

name = "level7.png"

hh, ww = img.shape[:2]

img = cv2.resize(img, (int(ww / 4), int(hh / 4)), interpolation=cv2.INTER_AREA)


img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
#img = cv2.medianBlur(img, 5)
B, G, R = cv2.split(img)
print(type(B))


img_exgr = calc_ExGR(B, G, R)
print(type(img_exgr))
show_image(img_exgr, "img_exgr")
img_exgr = img_exgr.astype(np.uint8)


th_e, exgr_bin = cv2.threshold(img_exgr, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
show_image(exgr_bin, "exgr_bin")

exgr_bin = cv2.morphologyEx(exgr_bin, cv2.MORPH_OPEN, kernelOpen)
exgr_bin = cv2.morphologyEx(exgr_bin, cv2.MORPH_CLOSE, kernelClose)
show_image(exgr_bin, "exgr_bin")
exgr_bin_not = cv2.bitwise_not(exgr_bin)

edges_canny = cv2.Canny(img, 60, 120)

edges_orig = structured_edge_bgr(img)
show_image(edges_orig, "edges_orig")


dist_map = cv2.distanceTransform(exgr_bin, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
show_image(dist_map, "dist_map")


img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv_structured = structured_edge(img_hsv)


H, S, V = cv2.split(img_hsv)
VVV = np.dstack([V, V, V])
VVV_structured = structured_edge(VVV)


exgr3 = np.dstack([img_exgr, img_exgr, img_exgr])
exgr_structured = structured_edge(exgr3)


orig_plus_exgr3 = edges_orig + exgr_structured
show_image(orig_plus_exgr3, "orig_plus_exgr3")
show_image(edges_orig, "edges_orig")
uh = orig_plus_exgr3.copy()
uh = (uh > 0.2) * uh
show_image(uh, "uh")



hh, ww = edges_orig.shape[:2]
print(hh)
print(ww)
# resize down, then back up

print(img.shape)




to_resize = uh.copy()
to_resize = to_resize * 255
to_resize = to_resize.astype(np.uint8)
show_image(to_resize, "to_resize")
#to_resize = cv2.bitwise_and(to_resize, to_resize, mask=exgr_bin_not)


rsize = 32
h = rsize
w = (ww/hh)*rsize
print("w v")
print(w)
result_0 = cv2.resize(to_resize, (int(w), int(h)), interpolation=cv2.INTER_AREA)
show_image(result_0, "pixelated_00")
result_0 = cv2.blur(result_0, (2, 2))
ret, result_0 = cv2.threshold(result_0, 10, 255, cv2.THRESH_BINARY)  #adaptiv?
result_0 = cv2.distanceTransform(result_0, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
show_image(result_0, "pixelated_0")
result_1 = cv2.resize(result_0, (ww, hh), interpolation=cv2.INTER_AREA)



neighborhood_size = 5
threshold = 0.005

data = result_0
show_image(result_0, "result_0")





x_min, y_min = locate_local_minimums(data, 5, 0.005)


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
    marker[int(y_min[i]), int(x_min[i])] = 1

show_image(marker, "marker")

ret, markers = cv2.connectedComponents(marker)


e = edges_orig.copy()
e = e * 255
e = e.astype(np.uint8)
show_image(e, "e")
edges_orig3 = np.dstack([e, e, e])
show_image(edges_orig3, "edges_orig3")
edges_orig3 = edges_orig3
wts_edge = cv2.watershed(edges_orig3, markers)


show_end()

show_image(wts_edge, "wts_edge")
wts_edge_rgb = wts_edge.copy()
wts_edge_rgb = wts_edge_rgb + 2
wts_edge_rgb = (255 / wts_edge_rgb.max()) * wts_edge_rgb
wts_edge_rgb = wts_edge_rgb.astype(np.uint8)
print(wts_edge_rgb)
wts_edge_rgb = cv2.cvtColor(wts_edge_rgb, cv2.COLOR_GRAY2RGB)
show_image(wts_edge_rgb, "wts_edge_rgb")
to_save = Image.fromarray(wts_edge_rgb)

to_save.save(name)


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


masked_V = cv2.bitwise_and(V, V, mask=maskClose)

masked_V = cv2.medianBlur(masked_V, 3)
edges = cv2.Canny(masked_V, 150, 200)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
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


plt.show()
