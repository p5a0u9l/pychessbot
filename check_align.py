import numpy as np
import matplotlib.pyplot as plt
from pyscreenshot import grab
from skimage.io import imsave
from skimage.feature import corner_foerstner, corner_peaks, corner_harris
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, rank
from skimage.morphology import opening, disk, closing
from skimage.feature import canny
from skimage.util import img_as_ubyte
import cv2


# x = 900
x = 1
y = 1
w = 1980
h = 1080
# h = 426
sq = w/8
# board lu list contains  =  x, y for each row, col address on the board
board_lu = [[(x + sq*r + sq/2, y + w - sq*c - sq/2) for r in range(8)]
            for c in range(8)]
board_lu = np.array(board_lu).ravel().reshape((64, 2))


def grab_board(x=x, y=y, w=w, h=h):
    # print "Grabbing board image at (%d, %d) x (%d, %d) ..." % (x, y, x + w, y + h),
    im = grab(bbox=(x, y, x + w, y + h), backend='scrot')
    im.save('.screenshot.png')
    return plt.imread('.screenshot.png')


def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1)*np.dot(d2, d2)))


def find_squares(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    gray = img_as_ubyte(rgb2gray(img))  # convert to gray scale
    selem = disk(100)
    im_bin = gray >= cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)  # global otsu binarize
    edges = canny(im_bin, sigma=.1)  # get edges from binary image
    edges = img_as_ubyte(edges)
    squares = []
    bin, contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cnt_len = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
        if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
            cnt = cnt.reshape(-1, 2)
            max_cos = np.max(
                [angle_cos(cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4]) for i in xrange(4)])
            if max_cos < 0.1:
                squares.append(cnt)
    return squares


def get_most_common_square(squares):
    area = np.zeros((len(squares,)))
    for i, sq in enumerate(squares):
        area[i] = cv2.contourArea(sq)
    area_ = np.int64(np.round(area))
    count = np.bincount(area_)
    return area, np.argmax(count)

im = grab_board()  # get image
squares = find_squares(img_as_ubyte(im))
area, square = get_most_common_square(squares)
# import ipdb; ipdb.set_trace()

im_sq = np.zeros(im.shape, np.uint8)
for sq in squares:
    for s in sq:
        im_sq[s[1], s[0], :] = 255


w, h, d = im.shape  # image dims
gray = rgb2gray(im)  # convert to gray scale
selem = disk(100)
im_bin = gray >= rank.threshold(gray, selem)  # global otsu binarize
edges = canny(im_bin, sigma=.1)  # get edges from binary image
corns = corner_harris(edges, sigma=.2)  # get corners
corns = corner_peaks(corns)
x = np.zeros(im.shape)
x[:, :, 0] += 255*edges
x[corns[:, 0] + 2, corns[:, 1] + 2, :] = 255
# for c in corns:

#     dist = np.abs(c - corns)
x = np.uint8(x)


# global otsu
plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1)
plt.imshow(im_bin, cmap="gray")
plt.title("Global Otsu binarization")

# Morph
plt.subplot(2, 2, 2)
plt.imshow(x)
plt.title("Eroded")


# Edge detection
plt.subplot(2, 2, 3)
plt.imshow(edges, cmap="gray")
plt.title("Edges of global bin")

# Corner detection
plt.subplot(2, 2, 4)
# plt.scatter(corns[:, 1], corns[:, 0], marker='.')
# plt.xlim((0, w-1))
# plt.ylim((0, h-1))
# plt.gca().invert_yaxis()
# plt.title("Foerstner Corner Detection on Global bin")
plt.imshow(im_sq)
plt.tight_layout()
plt.show()

imsave("bin.png", img_as_ubyte(im_bin))
imsave("sq.png", img_as_ubyte(im_sq))
imsave("edges.png", img_as_ubyte(edges))
imsave("x.png", x)
# imsave("corners.png", corns)
