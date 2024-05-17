# README #

## What is this repository for? ##

* Quick summary
	* A Python-based Dynamic multi-bed PET simulation and reconstruction framework with analytical modeling of system matrix, capable of incorporating various degrees of PSF modeling. 
* Version
	* 1.0.1.1

## How do I get set up? ##
* Necessary packages
	* NiBabel (To read and write NIfTI files)
	* scikit-image
	* SciPy
	* sklearn

* Start-up
	* To start the project you need to run the file **SimulateDynamicMulibedFDG.py**

* Configurations (config.json)
	* You can change various parameters in the config file such as:
		* Dimensions (xdim, ydim, zdim)
		* Number of frames
		* Axial and trans axial Field of View
		* Iterations
		* Subsets
		* Scan Duration
		* VCT Sensitivity
		* Attenuation units (if mu_map is in units of 1/voxel attenuation, that's the default assumption. If it's in units of 1/mm, then one needs to multiply it by bins-size)
	* Additional configurations can be added to automate and ease the process of simulation.

* Image Comparison
	* File **CompareOutputImages.py** calculates the following metrics between MATLAB and Python generated files:
		* Mean Squared Error (MSE)
	  	* Peak Signal to Noise Ratio (PSNR)
	  	* Structural Similarity Index (SSIM)
	* This file is run independently.


This code was developed by Nolla Sherifi (*nolla.sherifi@tum.de*) between October 2023 and March 2024 at the Technical University of Munich.