import ctypes
import math
import numpy as np

def generate_PSF_kernels(PSF_Kernel, xdim, SUBSETS, NUM_BINS, bin_size, scanner):
    if PSF_Kernel == 0:  # No PSF modeling
        Kernels2Select = [1]
    elif PSF_Kernel == 1:  # True PSF modeling
        Kernels2Select = [26]
    elif PSF_Kernel == 2:  # No PSF + True PSF modeling
        Kernels2Select = [1, 26]
    else:  # Generalized
        Kernels2Select = [1, 23, 26, 32, 40]

    if SUBSETS == 0:
        SUBSETS = 7

    NUMVAR = 50

    scatter_set = ((np.arange(1, 51) * 0.04) **1) * 3.27
    non_coll_fact_set = (np.arange(0, 50) * 0.04) * 0.0022
    mln = (np.arange(0, 5.1, 0.1)) * 0.4
    mu_set = create_mu_set(mln)

    D = scanner["ring_diameter"]
    r = D / 2
    np.random.seed()

    KernelsSet_hold = np.zeros((NUM_BINS, 15, NUMVAR))
    KernelsSet = np.zeros((NUM_BINS, NUM_BINS, NUMVAR))

    if (xdim * bin_size) >= D: #replaced NUM_bins * bin_size
        ctypes.windll.user32.MessageBoxW(0, 'Object FOV hits the scanner!!', 'Warning!!!', 0)

    for nrv in range(NUMVAR):
        mu = mu_set[nrv]
        scatter_fact = scatter_set[nrv]
        non_coll_fact = non_coll_fact_set[nrv]

        if nrv == math.ceil(NUMVAR/2 + 1):
            mu = 0.087
            scatter_fact = 3.27
            non_coll_fact = 0.0022

        convolFINAL = np.zeros((NUM_BINS, NUM_BINS))
        convolOVERALL_hold = np.zeros((NUM_BINS, 15))

        delta = 1 / 11
        x = np.round(np.arange(-7.5 + delta / 2, 7.5, delta), 4) * bin_size
        sizex = len(x)
        halfsizex = sizex // 2
        binsdimhalf = NUM_BINS // 2
        DRF = np.zeros(len(x))

        for bin in np.arange(NUM_BINS):
            source_pos = (bin - (NUM_BINS + 1) / 2) * bin_size
            angle = np.arcsin(source_pos / r)
            angle_deg = angle * 180/math.pi

            if angle == 0:
                DRF = np.double(x == 0)
            else:
                DRF = np.exp(-np.abs(mu * x / np.cos(angle) / np.sin(angle)))

            if source_pos > 0:
                DRF[halfsizex + 1:] = 0
            else:
                DRF[:halfsizex] = 0

            convol = np.convolve(DRF, DRF)
            convolDRF = convol[::2]

            FWHM_noncollin = non_coll_fact * (2 * r * np.cos(angle))

            FWHM_scatter = scatter_fact * np.cos(angle)

            FWHM = np.sqrt(FWHM_noncollin ** 2 + FWHM_scatter ** 2)
            sigma = FWHM / 2.355
            convol2 = np.exp(-(x ** 2 / sigma ** 2))

            convolOVERALL = np.convolve(convolDRF, convol2, mode='same')
            convolOVERALL = convolOVERALL / np.sum(convolOVERALL)
            convolOVERALL[np.isnan(convolOVERALL)] = 0

            for jjj in np.arange(15):
                if 0 <= bin + jjj - 8 < NUM_BINS:
                    start_index = int((jjj-1)/delta + 1)
                    end_index = int((jjj/delta))
                    convolOVERALL_hold[bin, jjj] = np.sum(convolOVERALL[start_index:end_index])
                    convolFINAL[bin, bin + jjj - 8] = np.sum(convolOVERALL[start_index:end_index])

        convolFINAL_orig = convolFINAL
        convolOVERALL_hold_orig = convolOVERALL_hold
        convolFINAL_tr = convolFINAL.transpose()

        if nrv == (NUMVAR // 2 + 1):
            KernelFull_hold = convolOVERALL_hold_orig
            KernelFull = convolFINAL

        KernelsSet_hold[:, :, nrv] = convolOVERALL_hold
        KernelsSet[:, :, nrv] = convolFINAL

    KernelsSet = KernelsSet[:, :, Kernels2Select]
    KernelsSet_hold = KernelsSet_hold[:, :, Kernels2Select]

    NUMVAR = len(Kernels2Select)

    return KernelFull_hold, KernelFull, KernelsSet_hold, KernelsSet, NUMVAR

def create_mu_set(mln):
    array1 = mln[25:34] * 0.087
    array2 = (mln[27:40] ** 7) * 0.087
    array3 = (mln[37::3] ** 7) * 0.087
    array4 = mln[1:26] * 0.087

    full_array = np.concatenate([array1, array2, array3, array4])
    unique_array = np.unique(full_array)
    unique_array.sort()
    unique_array = unique_array[::-1]

    return unique_array
