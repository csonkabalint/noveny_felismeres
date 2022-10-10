import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image

pi = math.pi

"""def points_in_circle(r, center_x, center_y, n=100):
    return [(math.cos((2*pi/n)*x)*r + center_x, math.sin((2*pi/n)*x)*r) + center_y for x in range(0, n+1)]"""


def points_in_circle(r, center_x, center_y, n=100):
    x_m, y_m = [], []
    for x in range(0, n + 1):
        x_m.append(math.cos((2*pi/n)*x)*r + center_x)
        y_m.append(math.sin((2*pi/n)*x)*r + center_y)
    return x_m, y_m


def is_inside(y_pos, x_pos, mask):
    for px, py in zip(x_pos, y_pos):
        print(px, py)
        print(mask[int(px)][int(py)])
        if mask[int(px)][int(py)] == 0:
            return False
    return True


def show_end():
    plt.show()
    exit(0)


def show_image(img_b, title):
    fig = plt.figure()
    fig.canvas.set_window_title(title)
    plt.imshow(img_b)


file_name = "level1"

img_orig = cv2.imread("imageProcessTestData/{}.jpg".format(file_name))
hh, ww = img_orig.shape[:2]
comp_rate = 4
img_orig_res = cv2.resize(img_orig, (int(ww / comp_rate), int(hh / comp_rate)), interpolation=cv2.INTER_AREA)

show_image(img_orig_res, "img_orig")

img = cv2.imread("{}_cut.png".format(file_name))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = img.astype(np.uint8)
img_dist = cv2.distanceTransform(img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
result = np.where(img_dist == img_dist.max())

r = 1
x_min, y_min = points_in_circle(r, result[1], result[0], 20)
while is_inside(x_min, y_min, img):
    r = r + 10
    x_min, y_min = points_in_circle(r, result[1], result[0], 20)
    #show_image(img, r)
    plt.autoscale(False)
    plt.plot(x_min, y_min, 'ro')

square_y = math.cos(5*pi/4)*r + result[0]
square_x = math.sin(5*pi/4)*r + result[1]
#square_x, square_y = points_in_circle(r - 10, result[1], result[0], 20)
side_length = math.sqrt(2) * r
print(square_y[0])
crop_img_res = img_orig_res[int(square_y[0]):int(square_y[0]) + int(side_length), int(square_x[0]):int(square_x[0]) + int(side_length)]
crop_img = img_orig[int(square_y[0] * comp_rate):int(square_y[0] * comp_rate) + int(side_length * comp_rate), int(square_x[0] * comp_rate):int(square_x[0] * comp_rate) + int(side_length * comp_rate)]

for p in x_min, y_min:
    print(p)
show_image(img, "img")
show_image(img_dist, "img_dist")


show_image(img, "data")
plt.autoscale(False)
plt.plot(x_min, y_min, 'ro')

show_image(img_dist, "data")
plt.autoscale(False)
plt.plot(result[1], result[0], 'ro')

show_image(img_orig_res, "data")
plt.autoscale(False)
plt.plot(square_x, square_y, 'ro')


show_image(crop_img_res, "crop_img_res")
show_image(crop_img, "crop_img")

to_save = Image.fromarray(crop_img)
to_save.save("imageProcessTestData\\{}_crop.png".format(file_name))


show_end()
