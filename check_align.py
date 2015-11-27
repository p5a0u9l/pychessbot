import numpy as np
import matplotlib.pyplot as plt
from pyscreenshot import grab
from skimage.feature import corner_foerstner, corner_peaks
from skimage.color import rgb2gray


# x = 900
x = 900
y = 100
w = 700
h = 700
# h = 426
sq = w/8
# board lu list contains  =  x, y for each row, col address on the board
board_lu = [[(x + sq*r + sq/2, y + w - sq*c - sq/2) for r in range(8)]
            for c in range(8)]
board_lu = np.array(board_lu).ravel().reshape((64, 2))


def grab_board():
    # print "Grabbing board image at (%d, %d) x (%d, %d) ..." % (x, y, x + w, y + h),
    im = grab(bbox=(x, y, x + w, y + h), backend='scrot')
    im.save('img.png')
    # print "Success."
    return plt.imread('img.png')


im = grab_board()


gray = rgb2gray(im)
plt.imshow(gray, cmap="gray"); plt.show()

x = corner_foerstner(gray, sigma=0.2)
pks = corner_peaks(x[1])
plt.scatter(pks[:, 1], pks[:, 0], marker='.'); plt.show()
