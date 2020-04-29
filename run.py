import random

import numpy as np
from scipy import ndimage
from skimage import io
from skimage.feature import canny
from skimage.filters import gabor_kernel

import matplotlib.pyplot as plt


def main():

    # 1. Canny edge detector
    ###

    data = image_data()[0]
    img = data['img_grayscale']

    # Extract contours from images
    edges = []
    for s in [1, 5, 10, 20]:
        pass

    # Repeat the same for all images:
    for data in image_data():
        img = data['img_grayscale']
        pass

    # 2. Filter bank
    ###

    data = image_data()[0]
    img = data['img_grayscale']

    # Visualize kernels
    kernels = create_filter_bank()
    kernel_selection = random.sample(kernels, k=6)

    pass

    # Apply them to a 1-channel image:
    pass


def create_filter_bank():
    """ Adapted from skimage doc. """
    kernels = []
    for theta in range(6):
        theta = theta / 4. * np.pi
        for sigma in (1, 3, 5):
            for frequency in (0.05, 0.15, 0.25):
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                              sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)
    return kernels

def apply_filter(image, kernel):
    return ndimage.convolve(image, kernel, mode='reflect')


def image_data():
    data = [
        {
            'name': 'IMD002.bmp',
            'position_lesion': [500, 300],
            'position_skin': [50, 50],
            # To be loaded in the following:
            'img': None,
            'img_grayscale': None,
            'markers': None
        },
        {
            'name': 'IMD004.bmp',
            'position_lesion': [500, 300],
            'position_skin': [50, 50],
            # To be loaded in the following:
            'img': None,
            'img_grayscale': None,
            'markers': None
        },
        {
            'name': 'IMD006.bmp',
            'position_lesion': [500, 300],
            'position_skin': [50, 50],
            # To be loaded in the following:
            'img': None,
            'img_grayscale': None,
            'markers': None
        }
    ]
    for d in data:
        name = d['name']
        d['img'] = io.imread(f'data/{name}').astype('float')
        d['img_grayscale'] = d['img'][:, :, 0]
        marker_lesion = get_marker(d['img_grayscale'], d['position_lesion'])
        marker_skin = get_marker(d['img_grayscale'], d['position_skin'])
        d['markers'] = marker_lesion + 2 * marker_skin

    return data


def get_marker(img, position):
    marker = np.empty_like(img)
    marker[position] = 1
    return marker


if __name__ == '__main__':
    main()
