import numpy as np
from scipy.signal import convolve2d
from skimage.transform import iradon


def gen_sens_sino(kernel_set, atten, norm_image, numvar, subsets, reconst_rm, theta_m, xdim, ydim, filter_tr, iradon_interp):
    binsdim, thdim_m, _ = atten.shape
    sensit_image_all = np.zeros((xdim, ydim, subsets, numvar))

    for nrv in range(0, numvar):
        convol_final_tr = np.transpose(np.squeeze(kernel_set[:, :, nrv]))

        for sub in range(0, subsets):
            proj = atten[:, :, sub] * norm_image[:, :, sub]
            if reconst_rm:
                proj2 = np.zeros((binsdim, thdim_m))
                for angle in range(0, thdim_m):
                    for binelm in range(0, binsdim):
                        proj2[:, angle] = proj2[:, angle] + proj[binelm, angle] * np.transpose(convol_final_tr[binelm, :])

            else:
                proj2 = proj

            aaa = iradon(proj2, theta_m[:, sub], filter_name=None, interpolation=iradon_interp, circle=False)

            if reconst_rm:
                sensit_image_all[:, :, sub, nrv] = convolve2d(aaa, filter_tr, mode='same')
            else:
                sensit_image_all[:, :, sub, nrv] = aaa

    return sensit_image_all
