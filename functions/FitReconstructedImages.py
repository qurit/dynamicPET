import numpy as np
import nibabel as nib
from sklearn.linear_model import LinearRegression
import os
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def regression(zz, total_frames, xsize, ysize, image, sp_list, cp_list):
	K = np.zeros((xsize, ysize))
	B = np.zeros((xsize, ysize))
	x = sp_list[:total_frames] / cp_list[:total_frames]
	X = np.column_stack((x, np.ones(total_frames)))
	model = LinearRegression()
	cp = cp_list[:total_frames]

	for xx in range(0, xsize):
		for yy in range(0, ysize):
			C = image[xx, yy, zz, :]
			y = C/cp
			model.fit(X, y)
			K[xx, yy], B[xx, yy] = model.coef_

	return K, B

def multicore_regression(args):
    return regression(*args)

def fitImages(total_frames, xsize, ysize, zsize, ITERATIONS, SUBSETS, output_path, Cp, Cp_integrated):
	Cp = np.array(Cp)
	Cp_integrated = np.array(Cp_integrated)
	image = np.zeros((xsize, ysize, zsize, total_frames))
	for frame in range(0, total_frames):
		fname = 'output_images_frame'+ str(frame + 1) + '_recon_it' + str(ITERATIONS) + '_subset' + str(SUBSETS) + '.nii'
		fpath = os.path.join(output_path, fname)
		outImage = nib.load(fpath).get_fdata()
		for zz in range(0, zsize):
			image[:,:,zz,frame] = outImage[:, :, zz]

	K_image = np.zeros((xsize, ysize, zsize))
	B_image = np.zeros((xsize, ysize, zsize))

	num_cores = os.cpu_count()
	chunksize = round(zsize/num_cores/5)
	args = [(zz, total_frames, xsize, ysize, image, Cp_integrated, Cp) for zz in range(0, zsize)]
	results = process_map(multicore_regression, args, max_workers=num_cores, chunksize=chunksize)
	K_image_slices, B_image_slices = zip(*results)
  
	for zz, img in enumerate(K_image_slices):
		K_image[:, :, zz] = img

	for zz, img in enumerate(B_image_slices):
		B_image[:, :, zz] = img

	finalized_K_image = nib.Nifti1Image(K_image, affine=np.eye(4))
	finalized_B_image = nib.Nifti1Image(B_image, affine=np.eye(4))

	filename_K = "output_images_recon_it{}_subset{}_K.nii".format(ITERATIONS, SUBSETS)
	filename_B = "output_images_recon_it{}_subset{}_B.nii".format(ITERATIONS, SUBSETS)

	filepath_K = os.path.join(output_path, filename_K)
	filepath_B = os.path.join(output_path, filename_B)

	nib.save(finalized_K_image, filepath_K)
	nib.save(finalized_B_image, filepath_B)
	return