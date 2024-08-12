import ctypes
import math
import os
import nibabel as nib
import numpy as np
import skimage.transform as st
from scipy.signal import convolve2d
from skimage.transform import radon, iradon
from functions.CalculateCalibrationFactor import calib_factor
from functions.GenerateSensitivitySinogram import gen_sens_sino
from functions.CalculateScalingFactor import scaling_factor

def perform_reconstruction(image_input, atten_input, ITERATIONS, SUBSETS, xdim, bin_size, voxel_size, d_z, ScanDuration, input_path, output_path, config, scanner, NUM_BINS, KernelFull, KernelsSet, NUMVAR):
    
    # Simulation Flags
    Num_Noise_Realz = config["Num_Noise_Realz"]
    NOISE_REALZ_Mean_Recon_Img = config["NOISE_REALZ_Mean_Recon_Img"]
    IMG_ABS_PRS = config["IMG_ABS_PRS"]
    RECON_NF_NOISY = config["RECON_NF_NOISY"]
    RECONST_RM = config["RECONST_RM"]
    SIMULATE_RM = config["SIMULATE_RM"]
    IMAGE_DECAYED = config["IMAGE_DECAYED"]
    HIGH_RES_TRUE = config["HIGH_RES_TRUE"]
    LOAD_ATTENUATION = config["LOAD_ATTENUATION"]
    LOAD_NORMALIZATION = config["LOAD_NORMALIZATION"]
    AOC_ind = config["AOC_ind"]
    AOC_unit = config["AOC_unit"]
    TOF = config["TOF"]

    VCT_sensitivity = scanner["VCT_sensitivity"] / 1e6
    start_time = config["start_time"]

    ydim = xdim
    input_xdim = xdim

    image_true = image_input
    image_correction = 1.0
    if AOC_ind == 1:
        image_correction = 1.0 / ((voxel_size/10)*(voxel_size/10)*(d_z/10))

    mu_map = np.zeros((xdim, ydim))
    if LOAD_ATTENUATION == 1:
        mu_map = atten_input

    thdim = math.pi / 2 * NUM_BINS
    thdim = (np.floor(thdim / SUBSETS) + 1) * SUBSETS
    d_theta = 180 / thdim
    thdim_m = int(thdim / SUBSETS)
    theta = np.arange(0, 180, d_theta)
    theta_m = theta.reshape((round(len(theta)/SUBSETS), SUBSETS))
    angles_m = len(theta_m)

    np.random.seed()
    random_numbers = [np.random.random() for _ in range(5)]
    random_numbers

    iRadonInterp = 'linear'

    HiResScale = input_xdim / xdim # to be updated - currently not used

    if not HIGH_RES_TRUE:
        if HiResScale != 1:
            ctypes.windll.user32.MessageBoxW(0, 'Fix HiRes Sizes!', 'Warning!!!', 0)
        HiResScale = 1

    if xdim == 128:
        try:
            if HiResScale == 4:
                OffsetCorr = -6
            elif HiResScale == 8:
                OffsetCorr = -13
        except:
            raise ValueError(
                "Sorry; The offset correction for your desired input image resolution has not been previously set. Please use the above few lines to define it. You may do some try and errors to find the best offset value.")
        BkgndBox = [41, 91, 32, 95]
    elif xdim == 256:
        try:
            if HiResScale == 4:
                OffsetCorr = -8
            elif HiResScale == 8:
                OffsetCorr = -13
        except:
            raise ValueError(
                "Sorry; The offset correction for your desired input image resolution has not been previously set. Please use the above few lines to define it. You may do some try and errors to find the best offset value.")
        BkgndBox = [1, 256, 1, 256]

    if LOAD_NORMALIZATION:
        if xdim == 128:
            norm_original_file = 'norm_map_128.nii'
            norm_original_filepath = os.path.join(input_path, norm_original_file)
            norm_original = nib.load(norm_original_filepath).get_fdata()
        elif xdim == 256:
            norm_original_file = 'norm_map_256.nii'
            norm_original_filepath = os.path.join(input_path, norm_original_file)
            norm_original = nib.load(norm_original_filepath).get_fdata()

        if norm_original.shape[0] * norm_original.shape[1] != NUM_BINS * thdim:
            norm_original = st.resize(norm_original, (NUM_BINS, int(thdim)), order=1, preserve_range=True)
    else:
        norm_original = np.ones((NUM_BINS, int(thdim)))

    Y2, X2 = np.meshgrid(np.arange(1, 7), np.arange(1, 7))
    filter = 1
    filter_tr = np.array([[filter]])

    if IMG_ABS_PRS == 0:
        ABS_PRS = [1]
    elif IMG_ABS_PRS == 1:
        ABS_PRS = [2]
    else:
        ABS_PRS = [1,2]

    if TOF:
        non_zero_area = np.count_nonzero(mu_map)*(bin_size*10**-3)**2
        equivalent_diameter = np.sqrt(4*non_zero_area/np.pi)
        TOF_factor = np.sqrt(2*equivalent_diameter/(scanner["TOF_resolution"]*10**-12)/299792458)
        if TOF_factor < 1:
            TOF_factor = 1
    else:
        TOF_factor = 1

    Recon_Img = np.zeros((xdim, xdim))
    MeanImg = np.zeros((xdim, ydim, ITERATIONS, NUMVAR))

    for realz in np.arange(Num_Noise_Realz):
        for sig in ABS_PRS:

            true_image_data = image_true * image_correction

            Y_bar0 = np.zeros((NUM_BINS, angles_m, SUBSETS))
            Y_bar = np.zeros((NUM_BINS, thdim_m, SUBSETS))

            atten = np.ones((NUM_BINS, thdim_m, SUBSETS))

            for sub in np.arange(SUBSETS):
                transform_angles = [angles[sub] for angles in theta_m]
                radon_transform = radon(mu_map, theta = transform_angles, circle=False)
                atten[:,:,sub] = np.exp(-radon_transform)

            norm_image = norm_original.reshape(NUM_BINS, thdim_m, SUBSETS)
            sensit_image_all = np.zeros((xdim, ydim, SUBSETS, NUMVAR))
            sensit_image_all = gen_sens_sino(KernelsSet, atten, norm_image, NUMVAR, SUBSETS, RECONST_RM, theta_m, xdim, ydim, filter_tr, iRadonInterp)

            lambda_val = 0.0063
            end_time = start_time + ScanDuration
            calibration_factor = calib_factor(xdim, ydim, VCT_sensitivity, theta_m, norm_image, NUM_BINS)
            scale_factor = scaling_factor(AOC_unit, voxel_size/HiResScale/10, voxel_size/HiResScale/10, d_z/10, start_time, end_time, lambda_val, calibration_factor, IMAGE_DECAYED, TOF_factor)

# See comment at end re: signal absent/present
            # SIG_ABS = np.zeros((len(range(BkgndBox[0], BkgndBox[1] + 1)), len(range(BkgndBox[2], BkgndBox[3] + 1)), ITERATIONS, NOISE_REALZ_Mean_Recon_Img, NUMVAR), dtype=np.float32)
            # SIG_ABS_NF = np.zeros((len(range(BkgndBox[0], BkgndBox[1] + 1)), len(range(BkgndBox[2], BkgndBox[3] + 1)), ITERATIONS, NOISE_REALZ_Mean_Recon_Img, NUMVAR), dtype=np.float32)
            # SIG_PRS = np.zeros((len(range(BkgndBox[0], BkgndBox[1] + 1)), len(range(BkgndBox[2], BkgndBox[3] + 1)), ITERATIONS, NOISE_REALZ_Mean_Recon_Img, NUMVAR), dtype=np.float32)
            # SIG_PRS_NF = np.zeros((len(range(BkgndBox[0], BkgndBox[1] + 1)), len(range(BkgndBox[2], BkgndBox[3] + 1)), ITERATIONS, NOISE_REALZ_Mean_Recon_Img, NUMVAR), dtype=np.float32)

            if SIMULATE_RM:
                image_true2 = convolve2d(true_image_data, filter_tr, mode='same')
            else:
                image_true2 = true_image_data

            for sub in np.arange(SUBSETS):
                transform_angles = [angles[sub] for angles in theta_m]
                radon_transform1 = radon(image_true2, transform_angles, circle=False)
                Y_bar0[:,:,sub] = radon_transform1

            Y_summed = np.zeros((NUM_BINS, angles_m, SUBSETS))

            if HiResScale != 1:
                for bin in range(2, NUM_BINS):
                    TmpInd = [((bin-1) * HiResScale + OffsetCorr + 1) , (bin * HiResScale + OffsetCorr)]
                    if (all(i > 0 for i in TmpInd) and all(j < Y_bar0.shape[0] for j in TmpInd)):
                        curr_bins = TmpInd
                        Y_summed[bin,:,:] = sum(Y_bar0[curr_bins,:,:], axis=0)
                Y_bar0 = Y_summed

            if SIMULATE_RM:
                for sub in range(0, SUBSETS):
                    for angle in range(0, thdim_m):
                        for bin in range(0, NUM_BINS):
                            Y_bar[:, angle, sub] = Y_bar[:, angle, sub] + Y_bar0[bin, angle, sub] * np.transpose(KernelFull[bin, :])
            else:
                Y_bar = Y_bar0

            Y_bar = Y_bar * atten * norm_image

            Y_bar_realization = np.zeros((NUM_BINS, thdim_m, SUBSETS, NOISE_REALZ_Mean_Recon_Img))

            if RECON_NF_NOISY == 0:
                NNF = [1]
            elif RECON_NF_NOISY == 1:
                NNF = [2]
            else:
                NNF = [1,2]

            for noisy in NNF:
                if noisy == 2:
                    Y_bar = Y_bar * scale_factor
                    for rlz in range(0, NOISE_REALZ_Mean_Recon_Img):
                        Y_bar_realization[:,:,:,rlz] = np.random.poisson(Y_bar)
                    Y_bar = Y_bar/scale_factor
                    Y_bar_realization = Y_bar_realization/scale_factor/(HiResScale**2)
                    T2 = 'Noisy'
                    tmp_NOISE_REALZ_Mean_Recon_Img = NOISE_REALZ_Mean_Recon_Img
                else:
                    for rlz in range(0, NOISE_REALZ_Mean_Recon_Img):
                        Y_bar_realization[:,:,:,rlz] = Y_bar/(HiResScale**2)
                    T2 = 'Noise Free'
                    tmp_NOISE_REALZ_Mean_Recon_Img = 1

                delta2 = 0.00000001
                unit_vector_image = np.ones((xdim,ydim))

                for nrv in range(0, NUMVAR):
                    convolFINAL = np.squeeze(KernelsSet[:,:,nrv])
                    convolFINAL_tr = np.transpose(convolFINAL)
                    for tmpRealz in range(0, tmp_NOISE_REALZ_Mean_Recon_Img):
                        # print('Reconstructing ' + T1 + ', ' + T2 + ', Realization #' + str(tmpRealz+1) + ' of ' + str(tmp_NOISE_REALZ_Mean_Recon_Img) + ', for PSF #' + str(nrv+1) + ' of ' + str(NUMVAR) + ' @ ' + str(time.time()) + ' sec')
                        image_old = unit_vector_image
                        for iter in range (0, ITERATIONS):
                            for sub in range(0, SUBSETS):
                                sensit_image = sensit_image_all[:,:,sub,nrv]
                                image_old2 = image_old
                                transform_angles = [angles[sub] for angles in theta_m]
                                data = radon(image_old2, transform_angles, circle=False)
                                if np.isnan(data).any():
                                    data[np.isnan(data)] = 0
                                if np.isinf(data).any():
                                    data[np.isinf(data)] = 0

                                if RECONST_RM:
                                    expected_data = np.zeros((NUM_BINS, thdim_m))
                                    for a1 in range(0, thdim_m):
                                        for b1 in range(0, NUM_BINS):
                                            expected_data[:,a1] = expected_data[:,a1] + data[b1,a1] * np.transpose(convolFINAL[b1,:])
                                else:
                                    expected_data = data

                                non_zero_exp_data = (expected_data > 0).astype(int)
                                zero_exp_data = (expected_data == 0).astype(int)
                                expected_data = expected_data + delta2 * zero_exp_data
                                ratio = non_zero_exp_data * Y_bar_realization[:,:,sub,tmpRealz]/expected_data

                                if RECONST_RM:
                                    ratio2 = np.zeros((NUM_BINS, thdim_m))
                                    for c1 in range(0, thdim_m):
                                        for d1 in range(0, NUM_BINS):
                                            ratio2[:, c1] = ratio2[:, c1] + ratio[d1, c1] * np.transpose(convolFINAL_tr[d1, :])
                                else:
                                    ratio2 = ratio

                                ratio2_inverseRadon = iradon(ratio2, theta_m[:,sub], filter_name=None, interpolation=iRadonInterp, circle=False)

                                if RECONST_RM:
                                    ratio2_inverseRadon = convolve2d(ratio2_inverseRadon, filter_tr, mode='same')

                                image_new = image_old * ratio2_inverseRadon / sensit_image
                                image_new[np.isnan(image_new)] = 0
                                image_old = image_new

# Related to signal absent/present. Saves image of only tumor, and only bakcground. Not relevenat for dynamic 3D imaging?
                            # if noisy == 2:
                            #     if sig == 1:
                            #         if SIG_ABS.shape == 2:
                            #             SIG_ABS[:,:] = image_new[BkgndBox[0]:BkgndBox[1],BkgndBox[2]:BkgndBox[3]]
                            #         if SIG_ABS.shape == 3:
                            #             SIG_ABS[:,:,iter] = image_new[BkgndBox[0]:BkgndBox[1],BkgndBox[2]:BkgndBox[3]]
                            #         if SIG_ABS.shape == 4:
                            #             SIG_ABS[:,:,iter,tmpRealz] = image_new[BkgndBox[0]:BkgndBox[1],BkgndBox[2]:BkgndBox[3]]
                            #         if SIG_ABS.shape == 5:
                            #             SIG_ABS[:,:,iter,tmpRealz,nrv] = image_new[BkgndBox[0]:BkgndBox[1],BkgndBox[2]:BkgndBox[3]]
                            #     elif sig == 2:
                            #         if SIG_PRS.shape == 2:
                            #             SIG_PRS[:,:] = image_new[BkgndBox[0]:BkgndBox[1],BkgndBox[2]:BkgndBox[3]]
                            #         if SIG_PRS.shape == 3:
                            #             SIG_PRS[:,:,iter] = image_new[BkgndBox[0]:BkgndBox[1],BkgndBox[2]:BkgndBox[3]]
                            #         if SIG_PRS.shape == 4:
                            #             SIG_PRS[:,:,iter,tmpRealz] = image_new[BkgndBox[0]:BkgndBox[1],BkgndBox[2]:BkgndBox[3]]
                            #         if SIG_PRS.shape == 5:
                            #             SIG_PRS[:,:,iter,tmpRealz,nrv] = image_new[BkgndBox[0]:BkgndBox[1],BkgndBox[2]:BkgndBox[3]]
                            # else:
                            #     if sig == 1:
                            #         if SIG_ABS_NF.shape == 2:
                            #             SIG_ABS_NF[:,:] = image_new[BkgndBox[0]:BkgndBox[1],BkgndBox[2]:BkgndBox[3]]
                            #         if SIG_ABS_NF.shape == 3:
                            #             SIG_ABS_NF[:,:,iter] = image_new[BkgndBox[0]:BkgndBox[1],BkgndBox[2]:BkgndBox[3]]
                            #         if SIG_ABS_NF.shape == 4:
                            #             SIG_ABS_NF[:,:,iter,tmpRealz] = image_new[BkgndBox[0]:BkgndBox[1],BkgndBox[2]:BkgndBox[3]]
                            #         if SIG_ABS_NF.shape == 5:
                            #             SIG_ABS_NF[:,:,iter,tmpRealz,nrv] = image_new[BkgndBox[0]:BkgndBox[1],BkgndBox[2]:BkgndBox[3]]
                            #     elif sig == 2:
                            #         if SIG_PRS_NF.shape == 2:
                            #             SIG_PRS_NF[:,:] = image_new[BkgndBox[0]:BkgndBox[1],BkgndBox[2]:BkgndBox[3]]
                            #         if SIG_PRS_NF.shape == 3:
                            #             SIG_PRS_NF[:,:,iter] = image_new[BkgndBox[0]:BkgndBox[1],BkgndBox[2]:BkgndBox[3]]
                            #         if SIG_PRS_NF.shape == 4:
                            #             SIG_PRS_NF[:,:,iter,tmpRealz] = image_new[BkgndBox[0]:BkgndBox[1],BkgndBox[2]:BkgndBox[3]]
                            #         if SIG_PRS_NF.shape == 5:
                            #             SIG_PRS_NF[:,:,iter,tmpRealz,nrv] = image_new[BkgndBox[0]:BkgndBox[1],BkgndBox[2]:BkgndBox[3]]

                            MeanImg[:,:,iter,nrv] = MeanImg[:,:,iter,nrv] + image_new

# Same as above
                # if noisy == 2:
                #     if sig == 1:
                #         Mean_SIG_ABS = MeanImg/NOISE_REALZ_Mean_Recon_Img
                #     elif sig == 2:
                #         Mean_SIG_PRS = MeanImg/NOISE_REALZ_Mean_Recon_Img
                # elif noisy == 1:
                #     if sig == 1:
                #         Mean_SIG_ABS = MeanImg/NOISE_REALZ_Mean_Recon_Img
                #         filename = 'SIG_ABS_NF__' + str(xdim) + '_R' + str(NOISE_REALZ_Mean_Recon_Img) + '_I' + str(ITERATIONS) + '_S' + str(SUBSETS) + '_K' + str(NUMVAR) + '_Tmr' + str(TmrCount) + '.nii'
                #         final_image = nib.Nifti1Image(Mean_SIG_ABS, affine=np.eye(4))
                #         nib.save(final_image, os.path.join(output_path, filename))
                #     elif sig == 2:
                #         Mean_SIG_PRS = MeanImg/NOISE_REALZ_Mean_Recon_Img
                #         filename = 'SIG_PRS_NF__' + str(xdim) + '_R' + str(NOISE_REALZ_Mean_Recon_Img) + '_I' + str(ITERATIONS) + '_S' + str(SUBSETS) + '_K' + str(NUMVAR) + '_Tmr' + str(TmrCount) + '.nii'
                #         final_image = nib.Nifti1Image(Mean_SIG_PRS, affine=np.eye(4))
                #         nib.save(final_image, os.path.join(output_path, filename))

        Recon_Img[:, :] = MeanImg[:,:,ITERATIONS-1, 0]/NOISE_REALZ_Mean_Recon_Img

        return Recon_Img
