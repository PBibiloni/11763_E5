import random

import numpy as np
from scipy import ndimage
from skimage import io
from skimage.feature import canny
from skimage.filters import gabor_kernel

import matplotlib.pyplot as plt


def main():
    data = image_data()[0]
    img = data['img_grayscale']

    # Extract contours from images
    edges = []
    for s in [1, 5, 10, 20]:
        edges.append(canny(img, sigma=s))

    fig, axs = plt.subplots(2, 3)
    fig.suptitle('Canny edge detector')
    axs[0][0].imshow(img)
    axs[0][1].imshow(edges[0])
    axs[0][2].imshow(edges[1])
    axs[1][1].imshow(edges[2])
    axs[1][2].imshow(edges[3])
    fig.show()

    # Repeat the same for all images:
    for data in image_data():
        img = data['img_grayscale']
        edgs = canny(img, sigma=15)

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Watershed on Canny edge detector')
        ax1.imshow(img)
        ax2.imshow(edgs)
        fig.show()

    # 2. Filter bank
    ###


    data = image_data()[0]
    img = data['img_grayscale']

    # Visualize kernels
    kernels = create_filter_bank()
    kernel_selection = random.sample(kernels, k=6)

    fig, axs = plt.subplots(2, 3)
    axs = [a for ax in axs for a in ax]
    fig.suptitle('Watershed on Canny edge detector')
    [ax.imshow(k) for k, ax in zip(kernel_selection, axs)]
    fig.show()

    # Apply them to a 1-channel image:
    fig, axs = plt.subplots(2, 4)
    axs = [a for ax in axs for a in ax]
    fig.suptitle('Watershed on Canny edge detector')
    axs[0].imshow(img)
    [ax.imshow(apply_filter(img, k)) for k, ax in zip(kernel_selection, axs[1:4] + axs[5:])]
    fig.show()


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
