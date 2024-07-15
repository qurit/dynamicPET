import numpy as np
from math import ceil
from skimage.transform import radon


def calib_factor(xdim, ydim, VCT_sens, theta_m, normalization, BIN_NUM):
    PSF_width = 0
    rod_img = np.zeros((xdim, ydim))
    TotalCounts = 1e9  # This number does not matter (do not touch); it cancels out
    rod_img[ceil(xdim/2), ceil(ydim/2)] = TotalCounts
    scaled_out = 0

    rod_sino = np.zeros(normalization.shape)
    for angle in np.arange(theta_m.shape[1]):
        radon_transform = radon(rod_img, theta=theta_m[:,angle - 1])
        pad_x = (rod_sino[:,:,angle - 1].shape[0] - radon_transform.shape[0])
        pad_y = 0
        pad_width = ((0, pad_x), (0, pad_y))
        rod_sino[:, :, angle - 1] = np.pad(radon_transform, pad_width, mode='constant', constant_values=0)

    rod_sino = rod_sino * normalization

    if np.sum(rod_sino) != 0:
        scaled_out = np.sum(rod_img) * VCT_sens / np.sum(rod_sino)

    return scaled_out
