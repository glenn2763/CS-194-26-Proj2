import numpy as np
import skimage as sk
import skimage.io as skio
import scipy
import matplotlib.pyplot as plt
from scipy import signal

im = skio.imread('data/skyline.jpg')

D_x = np.array([[-1, 1], [0, 0]], np.float32)
D_y = np.array([[1, 0], [-1, 0]], np.float32)

current_best = 0
best_angle = 0
for i in range(-10, 15):
    rotated = scipy.ndimage.interpolation.rotate(im[:,:,0], i, reshape=False)
    x = int(np.floor(len(rotated) * .25))
    y = int(np.floor(len(rotated[0]) * .25))
    rotated = rotated[x:-x, y:-y]
    rotated_x = signal.convolve2d(rotated, D_x, mode="same", boundary="symm")
    rotated_y = signal.convolve2d(rotated, D_y, mode="same", boundary="symm")
    grad = np.degrees(np.arctan2(-rotated_y, rotated_x))
    grad = grad.flatten()
    horiz = (np.abs(grad) < 5).sum()
    vert = (np.abs(np.abs(grad) - 90) < 5).sum()
    total_edges = horiz + vert
    if i == 0:
        plt.hist(np.abs(grad), bins=10)
        plt.show()
        print(total_edges)
    if i == 10: 
        plt.hist(np.abs(grad), bins=10)
        plt.show()
        print(total_edges)
    if total_edges > current_best:
        current_best = total_edges
        print(best_angle)
        best_angle = i

print(best_angle)
rotated = scipy.ndimage.interpolation.rotate(im, 10, reshape=False)
x = int(np.floor(len(rotated) * .1))
y = int(np.floor(len(rotated[0]) * .1))
best_image = rotated[x:-x, y:-y]

skio.imsave('docs/images/skyline_straight.jpg', best_image)

skio.imshow(best_image)
skio.show()