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
    	* "ITERATIONS": number of iterations for OSEM reconstruction
    	* "SUBSETS": number of subsets for OSEM reconstruction
    	* "mu_units": units of input MU map (/mm, /cm, or /voxel)
    	* "smoothing_kernel_fwhm" : post reconstruction smoothing kernel size (mm)
    	* "scanner": specified scanner model (see scanner_info.json for options)
    	* "kinetic_parameters_filename" : name of kinetic parameter input file (.json)
    	* "input_frame_durations" : duration of each frame of input function as comma delimited list (sec)
    	* "input_frame_starts" : start time of each frame as comma delimted list, frames considered to be successive if this is left blank (sec)
		* "input_function_concentration" : input function for kinetic modeling (any units)
		* "input_function_time" : input function time axis, overrides input_frame_durations if used (sec)
		* "output_frame_durations" : durations for simulated scan frames (sec), will use input_frame_durations & input_frame_starts if this left blank
		* "output_frame_starts" : start time of each frame as comma delimted list, frames considered to be successive if this is left blank (sec)

	* Simulation flags in config file (OFF:0, ON:1, unless otherwise specified)
    	* "LOAD_ATTENUATION" : enable attenuation correction from mu_map_file
    	* "LOAD_NORMALIZATION" : enable normalization (only availble for 128x128 & 256x256 images)
    	* "TOF" : enable correction factor to emulate reduced noise in TOF reconstruction
    	* "AOC_unit" : units of input function, 1 for [Bq/mL], 2 for [kBq/mL], 3 for [MBq/mL]
    	* "SMOOTHING" : enable post reconstruction smoothing, saved in addition to unsmoothed images
* Kinetic Parameters should be stored in kinetic_parameters.json in the following order: 'K1', 'k2', 'k3', 'k4', 'Vp'

## Authorship
The original simulation code was written by Saeed Ashrafinia in MATLAB for simulation of 2D PET frames for the purpose of evaluating PSF modeling (https://github.com/ashrafinia/PET_sim_recon).

A wrapper function for this 2D simulation was developed by Kyung-Nam Lee to enable simulation of dynamic PET (https://github.com/qurit/PET_Sim_Recon_Updated).

This MATLAB program was converted to python by Nolla Sherifi (*nolla.sherifi@tum.de*) at the Technical University of Munich.

Finally, the python code was refactored for performance and ease of use by James Fowler. Additional development was also done to improve the accuracy of the simulation, and allow for different imaging scenarios. This repository begins with changes implemented by James.