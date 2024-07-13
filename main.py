import time
import nibabel as nib
import numpy as np
from functions.GenerateCompartmentalImages import generate_graphics
from functions.MainPETSimulateReconstruct import perform_reconstruction
from functions.FitReconstructedImages import fitImages
import json
import os
from datetime import datetime
from tqdm import tqdm


def main():
    t0 = time.time()

    path = os.path.dirname(__file__)
    input_path = os.path.join(path, 'input')

    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H꞉%M꞉%S")
    output_path = os.path.join(path, 'output', now_str)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open("config.json", 'r') as f:
        config = json.load(f)
        
    ROIs_filename = config["ROIs_filename"]
    output_filename = config["output_filename"]
    mu_map_file = config["mu_map_file"]
    frames = config["frames"]
    ITERATIONS = config["ITERATIONS"]
    SUBSETS = config["SUBSETS"]
    ScanDuration = config["ScanDuration"]
    mu_units = config["mu_units"]

    with open(os.path.join(input_path, 'scanner_info.json'), 'r') as f:
        scanner_info = json.load(f)
    scanner = scanner_info[config["scanner"]]

    transaxial_FOV = scanner["transaxial_FOV"]
    axial_FOV = scanner["axial_FOV"]

    Background = [0.1, 1, 0, 0, 0.03]
    Bloodpool = [0, 1, 1, 1, 1]
    Myocardium = [0.6, 1.2, 0.1, 0.001, 0.05]
    Normal_Liver = [0.864, 0.981, 0.005, 0.016, 0]
    Normal_Lung = [0.108, 0.735, 0.016, 0.013, 0.017]
    Tumors_in_liver = [0.243, 0.78, 0.1, 0, 0]
    Tumors_in_lung = [0.044, 0.231, 1.149, 0.259, 0]
    values = [Background, Bloodpool, Myocardium, Normal_Liver, Normal_Lung, Tumors_in_liver, Tumors_in_lung]
    

    mu_map_filepath = os.path.join(input_path, mu_map_file)
    mu_map_3D = nib.load(mu_map_filepath).get_fdata()

    xdim, ydim, zdim = mu_map_3D.shape

    voxel_size = transaxial_FOV / xdim
    d_z = axial_FOV / (zdim - 1)

    final_image_3D = np.zeros((xdim, ydim, zdim))
    mu_map_slice = np.zeros((xdim, ydim))

    bin_size = transaxial_FOV / xdim

    if mu_units == '/mm':
        mu_map_3D = mu_map_3D * bin_size
    elif mu_units == '/cm':
        mu_map_3D = mu_map_3D * bin_size / 10
    elif mu_units != '/voxel':
        raise ValueError("mu_units must be 'mm', 'cm', or '/voxel'")
    
    ROIs_filepath = os.path.join(input_path, ROIs_filename)
    print('Generating Compartmental Images:')
    generate_graphics(values, ROIs_filepath, xdim, ydim, zdim, output_path)

    for frame in np.arange(frames):
        print("Simulating and Reconstructing Frame ", frame+1, ":")
        frn = int(frame) + 1
        frame_name = 'input_images_frame' + str(frn) + '.nii'
        frame_path = os.path.join(output_path, frame_name)
        frame_object = np.zeros((xdim, ydim, zdim))
        frame_object_slice = np.zeros((xdim, ydim))
        frame_object = nib.load(frame_path).get_fdata()

        for z in tqdm(np.arange(zdim)):
            mu_map_slice = mu_map_3D[:, :, z]
            frame_object_slice = frame_object[:,:,z]
            final_image_3D[:, :, z] = perform_reconstruction(frame_object_slice, mu_map_slice, ITERATIONS, SUBSETS, xdim, bin_size, voxel_size, d_z, ScanDuration, input_path, output_path, scanner)

        finalized_image = nib.Nifti1Image(final_image_3D, affine=np.eye(4))
        filename = "{}_frame{}_recon_it{}_subset{}.nii".format(output_filename, frame+1, ITERATIONS, SUBSETS)
        filepath = os.path.join(output_path, filename)
        nib.save(finalized_image, filepath)

    print("Fitting Reconstructed Images:")
    fitImages(frames, xdim, ydim, zdim, ITERATIONS, SUBSETS, output_path)

    #write log
    log_path = os.path.join(output_path, 'log.txt')
    with open(log_path, 'w') as f:
        f.write(f'{now_str}\n')
        for key, value in config.items():
            f.write(f'{key}: {value}\n')

    t1 = time.time()
    print("Time elapsed: ", t1 - t0)
    print("FDG Simulation Successful")

if __name__ == "__main__":
    main()
