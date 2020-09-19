# CS194-26 (CS294-26): Project 2

import numpy as np
import skimage as sk
import skimage.io as skio
import scipy
from skimage import feature
from skimage.filters import sobel
from scipy import signal
from scipy import ndimage


def gaussian_kernel(size=4, sigma=1):
    # algorithm taken from https://subsurfwiki.org/wiki/Gaussian_filter
    size = size // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def sobel_filters(img):
    # algorithm taken from University of Auckland 
    # https://www.cs.auckland.ac.nz/compsci373s1c/PatricesLectures/Edge%20detection-Sobel_2up.pdf
    horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    vertical = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    img_hor = ndimage.convolve(img, horizontal)
    img_ver = ndimage.convolve(img, vertical)
    
    G = np.hypot(img_hor, img_ver)
    G = G / np.amax(G) * 255
    theta = np.arctan2(img_ver, img_hor)
    
    return (G, theta)

def get_edges(img):
    smoothed = convolve(img, gaussian_kernel())
    sobel_filtered, _ = sobel_filters(smoothed)
    return sobel_filtered / np.amax(sobel_filtered)

def displace_image(img, displacement):
    rolled_x = np.roll(img, displacement[0], axis=0)
    return np.roll(rolled_x, displacement[1], axis=1)

# align the images
# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)

def align(image1, image2, displacement_range):
    # Take in two images and allign the first image onto the second

    # edge2 = sobel(image2) # Sobel filtering from skimage package
    # edge1 = sobel(image1)

    edge2 = get_edges(image2) # personal implementation of sobel filtering
    edge1 = get_edges(image1)

    # edge2 = image2 # using raw images to discern alignment
    # edge1 = image1

    lowest_diff = float('inf')
    for i in range(-displacement_range, displacement_range + 1):
        shifted_x = np.roll(edge1, i, axis=0)
        for j in range(-displacement_range, displacement_range + 1):
            shifted_xy = np.roll(shifted_x, j, axis=1)
            difference = shifted_xy - edge2
            total = sum(sum(difference * difference))
            if total < lowest_diff:
                best_i = i
                best_j = j
                lowest_diff = total
    displacement = np.array([best_i, best_j])
    return displacement

def pyramid_align(image1, image2, depth, scale_factor=0.5):
    if image1[0].size <= 500 or depth == 6:
        return align(image1, image2, 15)
    else:
        small1 = sk.transform.rescale(image1, scale_factor)
        small2 = sk.transform.rescale(image2, scale_factor)
        upwards_displacement = pyramid_align(small1, small2, depth + 1)
        upwards_displacement = upwards_displacement / scale_factor
        int_displacement = upwards_displacement.astype(int)
        close_image1 = displace_image(image1, int_displacement)
        displacement_range = 1 / scale_factor
        fine_tune = align(close_image1, image2, int(displacement_range))
        return int_displacement + fine_tune

def process_image(img):
    # read in the image
    im = skio.imread(img)

    # convert to double (might want to do this later on to save memory)    
    im = sk.img_as_float(im)

    # just grab the R channel from the image
    im_out = np.ndarray((len(im), len(im[0])))
    for i in range(len(im)):
        for j in range(len(im[0])):
            im_out[i][j] = im[i][j][0]

    # D_x = np.array([-1, 1], np.float32)

     
    # for i in range(len(im)):
    #     im_out[i] = ndimage.convolve1d(im_out[i], D_x) 

    D_x = np.array([[-1, 1], [0, 0]], np.float32)
    D_y = np.array([[1, 0], [-1, 0]], np.float32)

    im_out_x = signal.convolve2d(im_out, D_x, mode="same", boundary="symm")
    im_out_y = signal.convolve2d(im_out, D_y, mode="same", boundary="symm")

    im_out = np.sqrt(im_out_x * im_out_x + im_out_y * im_out_y)

    im_out = im_out / np.max(np.absolute(im_out))
    im_out = (im_out + 1) / 2.

    # # save the image
    # fname = 'output/' + img[5:len(img) - 4] + '.jpg'
    # skio.imsave(fname, im_out)

    # # display the image
    print(im_out.shape)
    skio.imshow(im_out)
    skio.show()

# name of the input file
imname = 'cameraman.png'

process_image('data/' + imname)