# README
## What is this repository for?

* Quick summary
	* A Python-based Dynamic multi-bed PET simulation and reconstruction framework with analytical modeling of system matrix, capable of incorporating various degrees of PSF modeling. 
* Version
	* 1.0.1.1

## How do I get set up?
* Necessary packages can be found in the env.yml file.

* Start-up
	* To start the project you need to run the file **SimulateDynamicMultibedFDG.py**

* Configurations (config.json)
	* All simulation parameters are in the config.json file:
		* "output_filename": name of output file
    	* "ROIs_filename": name of input ROI bitmask file (.nii)
    	* "mu_map_file": name of input MU map file (.nii)
    	* "frames": number of frames to be simulated
    	* "ITERATIONS": number of iterations for OSEM reconstruction
    	* "SUBSETS": number of subsets for OSEM reconstruction
    	* "mu_units": units of input MU map (/mm, /cm, or /voxel)
    	* "smoothing_kernel_fwhm" : post reconstruction smoothing kernel size (mm)
    	* "scanner": specified scanner model (see scanner_info.json for options)
    	* "kinetic_parameters_filename" : name of kinetic parameter input file (.json)
    	* "start_time" : scan start time (minutes)
    	* "frame_durations" : duration of frames (as array, in seconds)
	* Simulation flags in config file (OFF:0, ON:1, unless otherwise specified)
    	* "PSF_Kernel": 1,
    	* "Num_Noise_Realz" : 1,
    	* "NOISE_REALZ_Mean_Recon_Img" : 1,
    	* "IMG_ABS_PRS" : 0,
    	* "RECON_NF_NOISY" : 1,
    	* "RECONST_RM" : 1,
    	* "SIMULATE_RM" : 1,
    	* "IMAGE_DECAYED" : 0,
    	* "HIGH_RES_TRUE" : 0,
    	* "LOAD_ATTENUATION" : 1,
    	* "LOAD_NORMALIZATION" : 0,
    	* "TOF" : 1,
    	* "AOC_ind" : 2,
    	* "AOC_unit" : 1,
    	* "SMOOTHING" : post reconstruction smoothing

## Authorship
The original simulation code was written by Saeed Ashrafinia (JHU) in MATLAB for simulation of 2D PET frames for the purpose of evaluating PSF modeling.

A wrapper function for this 2D simulation was developed by Kyung Nam (UBC).

This MATLAB program was converted to python by Nolla Sherifi (*nolla.sherifi@tum.de*) at the Technical University of Munich.

Finally, the python code was refactored for performance and ease of use by James Fowler (UBC). Additional development was also done to improve the accuracy of the simulation, and allow for different imaging scenarios. This repository begins with changes implemented by James.

* Image Comparison
	* File **CompareOutputImages.py** calculates the following metrics between MATLAB and Python generated files:
		* Mean Squared Error (MSE)
	  	* Peak Signal to Noise Ratio (PSNR)
	  	* Structural Similarity Index (SSIM)
	* This file is run independently.