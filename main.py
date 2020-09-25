# CS194-26 (CS294-26): Project 2

import numpy as np
import skimage as sk
import skimage.io as skio
import scipy
import matplotlib.pyplot as plt
from scipy import signal


def gaussian_kernel(size=4, sigma=1.5):
    # algorithm taken from https://subsurfwiki.org/wiki/Gaussian_filter
    size = size // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

# kernel definitions
# gauss_kern = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 16.
gauss_kern = gaussian_kernel(6)

unit_impulse = signal.unit_impulse((7, 7), 'mid')
D_x = np.array([[-1, 1], [0, 0]], np.float32)
D_y = np.array([[1, 0], [-1, 0]], np.float32)

def sharpen(img, alpha=2.5):
    # read in the image
    im = skio.imread(img)

    # convert to double (might want to do this later on to save memory)    
    im = sk.img_as_float(im)
    
    # build an output image with same dimensions as input
    im_out = np.ndarray((len(im), len(im[0]), len(im[0][0])))

    # calculate sharpening kernel based on alpha
    sharp_kernel = (1 + alpha) * unit_impulse - alpha * gauss_kern

    # apply sharpening kernel to each channel and then add sharpened channel to output image
    for i in range(3):
        current_channel = im[:,:,i]
        sharp = signal.convolve2d(current_channel, sharp_kernel, mode='same', boundary='symm')
        im_out[:,:,i] = normalize(sharp)

    # fname = 'output/exsharp_' + img[7:len(img) - 4] + '.jpg'
    # skio.imsave(fname, im_out)
    skio.imshow(normalize(im + im_out))
    skio.show()
    return im_out

def blur(img, sigma=1.5):
    # read in the image
    im = skio.imread(img)

    # convert to double (might want to do this later on to save memory)    
    im = sk.img_as_float(im)

    # build an output image with same dimensions as input
    im_out = np.ndarray((len(im), len(im[0]), len(im[0][0])))

    # build a gaussian kernal from input paramaters
    gauss_kern = gaussian_kernel(6, sigma)

    for i in range(3):
        current_channel = im[:,:,i]
        im_out[:,:,i] = signal.convolve2d(current_channel, gauss_kern, mode='same', boundary='symm')
    
    # fname = 'output/exblur_' + img[5:len(img) - 4] + '.jpg'
    # skio.imsave(fname, im_out)

    return normalize(im_out)

def normalize(img):
    return (img - np.amin(img)) / (np.amax(img) - np.amin(img))


def get_edges(img):
    # read in the image
    im = skio.imread(img)

    # convert to double (might want to do this later on to save memory)    
    im = sk.img_as_float(im)

    # just grab the R channel from the image
    im_out = im[:,:,0]


    # gaussian blurring
    im_out = signal.convolve2d(im_out, gauss_kern, mode='same', boundary='symm')
    # im_out = signal.convolve2d(im_out, D_x, mode="same", boundary="symm")
    # im_out = signal.convolve2d(im_out, D_y, mode="same", boundary="symm")

    # Dx and Dy Kernel application
    im_out_x = signal.convolve2d(im_out, D_x, mode="same", boundary="symm")
    im_out_y = signal.convolve2d(im_out, D_y, mode="same", boundary="symm")
    im_out = np.sqrt(im_out_x * im_out_x + im_out_y * im_out_y)
    im_out = im_out / np.max(np.absolute(im_out))

    # combine convolution kernels
    # convx = signal.convolve2d(gauss_kern, D_x)
    # convtotal = signal.convolve2d(convx, D_y)
    # im_out = signal.convolve2d(im_out, convtotal)
    # im_out = im_out / np.max(np.absolute(im_out))
    # im_out = (im_out + 1) / 2.

    # make black and white at threshold
    for i in range(len(im_out)):
        for j in range(len(im_out[0])):
            if im_out[i][j] > 0.3:
                im_out[i][j] = 1
            else:
                im_out[i][j] = 0


    # # save the image
    # fname = 'docs/images/testx.jpg'
    # skio.imsave(fname, im_out_x)
    # fname = 'docs/images/testy.jpg'
    # skio.imsave(fname, im_out_y)

    # # display the image
    skio.imshow(normalize(im_out))
    skio.show()


# name of the input file
imname = 'cameraman.png'
# imname = 'facade.jpg'
# imname = 'taj.jpg'
# imname = 'mandelbrot.jpg'
# imname = 'nutmeg.jpg'

# function to run (choose one)
get_edges('data/' + imname)
# sharpen('data/' + imname)
