import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import skimage.io as skio
import scipy
from scipy import signal
from align_image_code import align_images



def gaussian_kernel(size=4, sigma=1.5):
    # algorithm taken from https://subsurfwiki.org/wiki/Gaussian_filter
    size = size // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def normalize(img):
    return (img - np.amin(img)) / (np.amax(img) - np.amin(img))

# First load images

# high sf
im1 = plt.imread('data/DerekPicture.jpg')/255.

# low sf
im2 = plt.imread('data/nutmeg.jpg')/255

# Next align images (this code is provided, but may be improved)
im1_aligned, im2_aligned = align_images(im2, im1)

## You will provide the code below. Sigma1 and sigma2 are arbitrary 
## cutoff values for the high and low frequencies


def hybrid_image(im1, im2, sigma1, sigma2):

    low_pass1 = np.ndarray((len(im1), len(im1[0]), len(im1[0][0])))
    low_pass2 = np.ndarray((len(im1), len(im1[0]), len(im1[0][0])))

    # build a gaussian kernel from input paramaters
    gauss_kern1 = gaussian_kernel(20, sigma1)
    gauss_kern2 = gaussian_kernel(20, sigma2)


    for i in range(3):
        current_channel_im1 = im1[:,:,i]
        current_channel_im2 = im2[:,:,i]
        low_pass1[:,:,i] = signal.convolve2d(current_channel_im1, gauss_kern1, mode='same', boundary='symm')
        low_pass2[:,:,i] = signal.convolve2d(current_channel_im2, gauss_kern2, mode='same', boundary='symm')

    high_pass2 = im2 - normalize(low_pass2)

    

    low_pass1 = normalize(low_pass1)
    high_pass2 = normalize(high_pass2)


    # cat_freq = np.log(np.abs(np.fft.fftshift(np.fft.fft2(im1[:,:,0]))))
    # derek_freq = np.log(np.abs(np.fft.fftshift(np.fft.fft2(im2[:,:,0]))))
    # lowpass_freq = np.log(np.abs(np.fft.fftshift(np.fft.fft2(low_pass1[:,:,0]))))
    # highpass_freq = np.log(np.abs(np.fft.fftshift(np.fft.fft2(high_pass2[:,:,0]))))
    # hybrid_freq = np.log(np.abs(np.fft.fftshift(np.fft.fft2(normalize(low_pass1 + high_pass2)[:,:,0]))))
    # cat_name = 'cat_freq.jpg'
    # skio.imsave(cat_name, cat_freq)
    # derek_name = 'derek_freq.jpg'
    # skio.imsave(derek_name, derek_freq)    
    # lowpass_name = 'lowpass_freq.jpg'
    # skio.imsave(lowpass_name, lowpass_freq)
    # highpass_name = 'highpass_freq.jpg'
    # skio.imsave(highpass_name, highpass_freq)
    # hybrid_name = 'hybrid_freq.jpg'
    # skio.imsave(hybrid_name, hybrid_freq)


    return normalize(low_pass1 + high_pass2)

sigma1 = 50
sigma2 = 50
hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)
skio.imshow(hybrid)
skio.show()

## Compute and display Gaussian and Laplacian Pyramids
## You also need to supply this function
N = 5 # suggested number of pyramid levels (your choice)

def pyramids(img, levels):
    if levels == 1:
        return img
    # for big hybrid images
    # gauss_kern = gaussian_kernel(10, 40)

    # for lincoln
    gauss_kern = gaussian_kernel(5, 10)


    gauss_pyramid = np.ndarray((len(img), len(img[0]), levels))
    laplace_pyramid = np.ndarray((len(img), len(img[0]), levels))
    gauss_pyramid[:,:,0] = img
    for i in range(1, levels):
        img = signal.convolve2d(img, gauss_kern, mode='same', boundary='symm')
        gauss_pyramid[:,:,i] = normalize(img)
    for i in range(0, levels - 1):
        laplace_pyramid[:,:,i] = normalize(gauss_pyramid[:,:,i] - gauss_pyramid[:,:, i + 1])
    laplace_pyramid[:,:,levels - 1] = normalize(gauss_pyramid[:,:,levels-1] - normalize(signal.convolve2d(gauss_pyramid[:,:,levels-1], gauss_kern, mode='same', boundary='symm')))
    return gauss_pyramid, laplace_pyramid


tree_test = plt.imread('docs/images/einstein_tree.jpg')/255.

im_out_gauss, im_out_laplace = pyramids(tree_test[:,:,0], 4)




# im_out, _ = pyramids(tree_test[:,:,0], 4)
# skio.imshow(im_out[:,:,0])
# skio.show()
# skio.imshow(im_out[:,:,1])
# skio.show()
# skio.imshow(im_out[:,:,2])
# skio.show()
# skio.imshow(im_out[:,:,3])
# skio.show()
