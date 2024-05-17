import time
import nibabel as nib
import numpy as np
import GenerateCompartmentalImages
import MainPETSimulateReconstruct
import FitReconstructedImages
import json


def main_simulate_reconstruct():
    t0 = time.time()

    with open("config.json", 'r') as f:
        config = json.load(f)
        
    ROIs_filename = config["ROIs_filename"]
    output_filename = config["output_filename"]
    mu_map_file = config["mu_map_file"]
    xdim = config["xdim"]
    ydim = xdim
    zdim = config["zdim"]
    frames = config["frames"]
    transaxial_FOV = config["transaxial_FOV"]
    voxel_size = transaxial_FOV / xdim
    axial_FOV = config["axial_FOV"]
    d_z = axial_FOV / (zdim - 1)
    ITERATIONS = config["ITERATIONS"]
    SUBSETS = config["SUBSETS"]
    ScanDuration = config["ScanDuration"]
    # VCT_sensitivity = 7300/1e6
    # VCT_sensitivity = 13300/1e6
    VCT_sensitivity = config["VCT_sensitivity"] / 1e6
    mu_units = config["mu_units"]

    Background = [0.1, 1, 0, 0, 0.03]
    Bloodpool = [0, 1, 1, 1, 1]
    Myocardium = [0.6, 1.2, 0.1, 0.001, 0.05]
    Normal_Liver = [0.864, 0.981, 0.005, 0.016, 0]
    Normal_Lung = [0.108, 0.735, 0.016, 0.013, 0.017]
    Tumors_in_liver = [0.243, 0.78, 0.1, 0, 0]
    Tumors_in_lung = [0.044, 0.231, 1.149, 0.259, 0]
    values = [Background, Bloodpool, Myocardium, Normal_Liver, Normal_Lung, Tumors_in_liver, Tumors_in_lung]

    final_image_3D = np.zeros((xdim, ydim, zdim))
    mu_map_3D = np.zeros((xdim, ydim, zdim))
    mu_map_slice = np.zeros((xdim, ydim))

    mu_map_3D = nib.load('NCAT_128x128x47_0347x0347x0327_atn_invmm.nii').get_fdata()

    bin_size = transaxial_FOV / xdim
    if mu_units == 1:
        mu_map_3D = mu_map_3D * bin_size

    GenerateCompartmentalImages.generate_graphics(values, ROIs_filename, xdim, ydim, zdim)

    for frame in np.arange(frames):
        print("Reconstructing frame: ", frame+1)
        frn = int(frame) + 1
        frame_name = 'input_images_frame' + str(frn) + '.nii'
        frame_object = np.zeros((xdim, ydim, zdim))
        frame_object_slice = np.zeros((xdim, ydim))
        frame_object = nib.load(frame_name).get_fdata()

        for z in np.arange(zdim):
            mu_map_slice = mu_map_3D[:, :, z]
            frame_object_slice = frame_object[:,:,z]
            final_image_3D[:, :, z] = MainPETSimulateReconstruct.perform_reconstruction(frame_object_slice, mu_map_slice, ITERATIONS, SUBSETS, xdim, bin_size, voxel_size, d_z, ScanDuration)

        finalized_image = nib.Nifti1Image(final_image_3D, affine=np.eye(4))
        filename = "output_images_frame{}_recon_it{}_subset{}.nii".format(frame+1, ITERATIONS, SUBSETS)
        nib.save(finalized_image, filename)

    FitReconstructedImages.fitImages(frames, xdim, ydim, zdim, ITERATIONS, SUBSETS)

    t1 = time.time()
    print("Time elapsed: ", t1 - t0)
    print("FDG Simulation Successful")
    input()

main_simulate_reconstruct()
