import numpy as np

def main():
    # Extract contours from image

    # Apply watershed

    # Visualize kernels

    # Apply them to the image




def create_filter_bank():
    """ Adapted from skimage doc. """
    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for frequency in (0.05, 0.25):
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                              sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)
    return kernels


if __name__ == '__main__':
    main()
