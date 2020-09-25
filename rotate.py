import numpy as np
import skimage as sk
import skimage.io as skio
import scipy
import matplotlib.pyplot as plt
from scipy import signal

im = skio.imread('data/facade.jpg')

D_x = np.array([[-1, 1], [0, 0]], np.float32)
D_y = np.array([[1, 0], [-1, 0]], np.float32)

current_best = 0
best_angle = 0
for i in range(-20, 20):
    rotated = scipy.ndimage.interpolation.rotate(im[:,:,0], i, reshape=False)
    x = int(np.floor(len(rotated) * .25))
    y = int(np.floor(len(rotated[0]) * .25))
    rotated = rotated[x:-x, y:-y]
    rotated_x = signal.convolve2d(rotated, D_x, mode="same", boundary="symm")
    rotated_y = signal.convolve2d(rotated, D_y, mode="same", boundary="symm")
    grad = np.degrees(np.arctan2(-rotated_y, rotated_x))
    grad = grad.flatten()
    # find and sum all the gradients that are less than 2 degrees from horizontal or vertical
    horiz = (np.abs(grad) < 2).sum()
    vert = (np.abs(np.abs(grad) - 90) < 2).sum()
    total_edges = horiz + vert
    if total_edges > current_best:
        current_best = total_edges
        best_angle = i

rotated = scipy.ndimage.interpolation.rotate(im, best_angle, reshape=False)
x = int(np.floor(len(rotated) * .1))
y = int(np.floor(len(rotated[0]) * .1))
best_image = rotated[x:-x, y:-y]

skio.imshow(best_image)
skio.show()