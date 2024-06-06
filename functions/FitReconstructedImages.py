import numpy as np
import nibabel as nib
from sklearn.linear_model import LinearRegression
import os
from tqdm import tqdm

def fitImages(total_frames, xsize, ysize, zsize, ITERATIONS, SUBSETS, output_path, working_path):
	sp_list_filename = os.path.join(working_path, 'sp_NP6.txt')
	cp_list_filename = os.path.join(working_path, 'cp_NP6.txt')

	sp_list = np.loadtxt(sp_list_filename)
	cp_list = np.loadtxt(cp_list_filename)

	image = np.zeros((xsize, ysize, zsize, total_frames))
	for frame in range(0, total_frames):
		fname = 'output_images_frame'+ str(frame + 1) + '_recon_it' + str(ITERATIONS) + '_subset' + str(SUBSETS) + '.nii'
		fpath = os.path.join(output_path, fname)
		outImage = nib.load(fpath).get_fdata()
		for zz in range(0, zsize):
			image[:,:,zz,frame] = outImage[:, :, zz]

	total_image_counts = np.squeeze(np.sum(image, axis=(0, 1, 2)))

	K_image = np.zeros((xsize, ysize, zsize))
	B_image = np.zeros((xsize, ysize, zsize))

	for xx in tqdm(range(0, xsize)):
		for yy in range(0, ysize):
			for zz in range(0, zsize):
				C = np.squeeze(image[xx, yy, zz, :])

				y = C/cp_list
				x = sp_list/cp_list

				Ones = np.ones(total_frames)

				X = np.zeros((6, 2))
				for i in np.arange(0, total_frames):
					X[i,:] = [x[i], Ones[i]]
				
				model = LinearRegression()
				model.fit(X, y)
				b = model.coef_

				K_image[xx, yy, zz] = b[0]
				B_image[xx, yy, zz] = b[1]

	finalized_K_image = nib.Nifti1Image(K_image, affine=np.eye(4))
	finalized_B_image = nib.Nifti1Image(B_image, affine=np.eye(4))

	filename_K = "output_images_recon_it{}_subset{}_K.nii".format(ITERATIONS, SUBSETS)
	filename_B = "output_images_recon_it{}_subset{}_B.nii".format(ITERATIONS, SUBSETS)

	filepath_K = os.path.join(output_path, filename_K)
	filepath_B = os.path.join(output_path, filename_B)

	nib.save(finalized_K_image, filepath_K)
	nib.save(finalized_B_image, filepath_B)