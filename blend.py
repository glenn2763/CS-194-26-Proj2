import numpy as np
import skimage as sk
import skimage.io as skio
import scipy
import matplotlib.pyplot as plt
from scipy import signal
from main import gaussian_kernel, normalize

im1 = skio.imread('data/apple.jpeg')/255.
im2 = skio.imread('data/orange.jpeg')/255.

gauss_kern = gaussian_kernel(5, 5)

def step_stack(im1, levels):
    step = np.ndarray((len(im1), len(im1[0]), levels))
    for i in range(len(step)):
        for j in range(len(step[0])):
            if j > len(step[0]) / 2:
                step[i][j][0] = 1
            else:
                step[i][j][0] = 0
    step[:,:,0] = normalize(signal.convolve2d(step[:,:,0], gaussian_kernel(20, 20), mode='same', boundary='symm'))
    for i in range(1, levels):
        step[:,:,i] = normalize(signal.convolve2d(step[:,:,i-1], gaussian_kernel(20, 20), mode='same', boundary='symm'))
    return step

def blend(im1, im2, levels):
    # building the gaussian blurred step function
    blended_levels = np.ndarray((len(im1), len(im1[0]), levels))
    output_image = np.ndarray((len(im1), len(im1[0])))
    step = step_stack(im1, levels)
    for i in range(len(step)):
        for j in range(len(step[0])):
            output_image[i][j] = 0

    
    # building laplacian stack for im1
    gauss_im1 = np.ndarray((len(im1), len(im1[0]), levels))
    laplace_im1 = np.ndarray((len(im1), len(im1[0]), levels))
    gauss_im1[:,:,0] = im1
    for i in range(1, levels):
        im1 = signal.convolve2d(im1, gauss_kern, mode='same', boundary='symm')
        gauss_im1[:,:,i] = normalize(im1)
    for i in range(0, levels - 1):
        laplace_im1[:,:,i] = normalize(gauss_im1[:,:,i] - gauss_im1[:,:,i+1])
    laplace_im1[:,:,levels - 1] = normalize(gauss_im1[:,:,levels-1] - normalize(signal.convolve2d(gauss_im1[:,:,levels-1], gauss_kern, mode='same', boundary='symm')))
    
    # building laplacian stack for im2
    gauss_im2 = np.ndarray((len(im2), len(im2[0]), levels))
    laplace_im2 = np.ndarray((len(im2), len(im2[0]), levels))
    gauss_im2[:,:,0] = im2
    for i in range(1, levels):
        im2 = signal.convolve2d(im2, gauss_kern, mode='same', boundary='symm')
        gauss_im2[:,:,i] = normalize(im2)
    for i in range(0, levels - 1):
        laplace_im2[:,:,i] = normalize(gauss_im2[:,:,i] - gauss_im2[:,:,i+1])
    laplace_im2[:,:,levels - 1] = normalize(gauss_im2[:,:,levels-1] - normalize(signal.convolve2d(gauss_im2[:,:,levels-1], gauss_kern, mode='same', boundary='symm')))
    
    for i in range(levels):
        blended_levels[:,:,i] = step[:,:,i] * laplace_im1[:,:,i] + (1 - step[:,:,i]) * laplace_im2[:,:,i]

    for i in range(levels):
        output_image = output_image + blended_levels[:,:,i]
    
    skio.imsave('docs/images/blended_level1.jpg', blended_levels[:,:,0])
    skio.imsave('docs/images/blended_level2.jpg', blended_levels[:,:,2])
    skio.imsave('docs/images/blended_level3.jpg', blended_levels[:,:,3])
    return normalize(output_image)

apple = im1[:,:,0]
orange = im2[:,:,0]
blended = blend(apple, orange, 4)



