import time
import nibabel as nib
import numpy as np
import math
from functions.GenerateCompartmentalImages import generate_graphics
from functions.GeneratePSFKernels import generate_PSF_kernels
from functions.MainPETSimulateReconstruct import perform_reconstruction
from functions.FitReconstructedImages import fitImages
import json
import os
from datetime import datetime
from tqdm.contrib.concurrent import process_map
from nilearn.image import smooth_img

def multicore_recon(args):
    return perform_reconstruction(*args)

def main_simulate():
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
        
    log_path = os.path.join(output_path, 'log.txt')
    with open(log_path, 'w') as f:
        f.write(f'{now_str}\n')
        for key, value in config.items():
            f.write(f'{key}: {value}\n')
        
    ROIs_filename = config["ROIs_filename"]
    output_filename = config["output_filename"]
    mu_map_file = config["mu_map_file"]
    ITERATIONS = config["ITERATIONS"]
    SUBSETS = config["SUBSETS"]
    input_frame_durations = [frame/60 for frame in config["input_frame_durations"]]
    output_frame_durations = [frame/60 for frame in config["output_frame_durations"]]
    mu_units = config["mu_units"]
    kinetic_parameters_filename = config["kinetic_parameters_filename"]
    smoothing_kernel_fwhm = config["smoothing_kernel_fwhm"]
    PSF_Kernel = 1 #config["PSF_Kernel"] 
    SMOOTHING = config["SMOOTHING"]
    LOAD_NORMALIZATION = config["LOAD_NORMALIZATION"]

    if not output_frame_durations:
        output_frame_durations = input_frame_durations

    frames = len(output_frame_durations)

    with open(os.path.join(input_path, 'scanner_info.json'), 'r') as f:
        scanner_info = json.load(f)
    scanner = scanner_info[config["scanner"]]

    transaxial_FOV = scanner["transaxial_FOV"]
    axial_FOV = scanner["axial_FOV"]
    
    with open(os.path.join(input_path, kinetic_parameters_filename), 'r') as file:
        parameters = json.load(file)
    kinetic_parameters = [value for key, value in parameters.items()]
    organs = [key for key, value in parameters.items()]

    mu_map_filepath = os.path.join(input_path, mu_map_file)
    mu_map_3D = nib.load(mu_map_filepath).get_fdata()
    mu_map_3D = mu_map_3D[::-1, ::-1, :]

    xdim, ydim, zdim = mu_map_3D.shape

    if LOAD_NORMALIZATION:
        if xdim != 128 or xdim !=256:
            print("Normalization only available for 128x128 or 256x256 images. Continuing without normalization.")

    voxel_size = transaxial_FOV / xdim
    d_z = axial_FOV / (zdim - 1)

    final_image_3D = np.zeros((xdim, ydim, zdim))

    bin_size = transaxial_FOV / xdim
    NUM_BINS = math.ceil(np.sqrt(2) * xdim)

    if mu_units == '/mm':
        mu_map_3D = mu_map_3D * bin_size
    elif mu_units == '/cm':
        mu_map_3D = mu_map_3D * bin_size / 10
    elif mu_units != '/voxel':
        raise ValueError("mu_units must be 'mm', 'cm', or '/voxel'")
    
    ROIs_filepath = os.path.join(input_path, ROIs_filename)
    print('Generating Compartmental Images:')
    Cp, Cp_integrated, output_frame_starts = generate_graphics(output_path, config, kinetic_parameters, ROIs_filepath, xdim, ydim, zdim)

    KernelFull_hold, KernelFull, KernelsSet_hold, KernelsSet, NUMVAR = generate_PSF_kernels(PSF_Kernel, xdim, SUBSETS, NUM_BINS, bin_size, scanner)

    for frame in np.arange(frames):
        print("Simulating and Reconstructing Frame", frame+1, ":")
        frn = int(frame) + 1
        frame_name = 'input_images_frame' + str(frn) + '.nii'
        frame_path = os.path.join(output_path, frame_name)
        frame_object = np.zeros((xdim, ydim, zdim))
        frame_object = nib.load(frame_path).get_fdata()

        num_cores = os.cpu_count()
        chunksize = round(zdim/num_cores/5)
        if not chunksize:
            chunksize = 1
        args = [(frame_object[:,:,z], mu_map_3D[:, :, z], ITERATIONS, SUBSETS, xdim, bin_size, voxel_size, d_z, output_frame_durations[int(frame)], input_path, output_path, config, scanner, NUM_BINS, KernelFull, KernelsSet, NUMVAR, output_frame_starts[int(frame)]) for z in np.arange(zdim)]
        final_image_3D_slices = process_map(multicore_recon, args, max_workers=num_cores, chunksize=chunksize)

        final_image_3D = np.zeros((xdim, ydim, zdim), dtype=np.float32)

        for z, img in enumerate(final_image_3D_slices):
            final_image_3D[:, :, z] = img

        finalized_image = nib.Nifti1Image(final_image_3D, affine=np.eye(4))
        filename = "{}_frame{}_recon_it{}_subset{}.nii".format(output_filename, frame+1, ITERATIONS, SUBSETS)
        filepath = os.path.join(output_path, filename)
        nib.save(finalized_image, filepath)
        
        if SMOOTHING:
            smooth_image = smooth_img(finalized_image, smoothing_kernel_fwhm)
            filename = "{}_frame{}_recon_it{}_subset{}_smooth.nii".format(output_filename, frame+1, ITERATIONS, SUBSETS)
            filepath = os.path.join(output_path, filename)
            nib.save(smooth_image, filepath)

    print("Fitting Reconstructed Images:")
    fitImages(frames, xdim, ydim, zdim, ITERATIONS, SUBSETS, output_path, Cp, Cp_integrated)

    t1 = time.time()
    elapsed_time = t1 - t0
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    with open(log_path, 'a') as f:
        f.write("Time elapsed: {:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds)))

    print("Time elapsed: {:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds)))
    print("FDG Simulation Successful")
    return

if __name__ == "__main__":
    main_simulate()
