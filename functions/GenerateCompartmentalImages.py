import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import convolve2d
from sklearn.linear_model import LinearRegression
import os
from tqdm import tqdm

def generate_graphics(values, ROIs_filename, xdim, ydim, zdim, output_path):
	sp_list_filename = os.path.join(output_path, 'sp_NP6.txt')
	cp_list_filename = os.path.join(output_path, 'cp_NP6.txt')

	plot_shape0 = ["b","r","g","c","k","b","r","g","c","k","b"]
	plot_shape1 = [":","--","-.","-","-","--","-.","-",":","--"]
	plot_shape2 = ["o","x","+","*","<","d","o","x","+","*","<","d"]

	plot_figures = 0

	if plot_figures:
		plt.plot()
		plt.show()

	N0 = 24
	N1 = 6
	N_frames = N0 + N1

	t_begin = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 553.6, 959.6, 1365.6, 1771.6, 2177.6, 2583.6]
	t_begin = [number/60 for number in t_begin]
	t_end = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 598.6, 1004.6, 1410.6, 1816.6, 2222.6, 2628.6]
	t_end = [number/60 for number in t_end]

	size_t = len(t_end)
	delta_t = []
	t_mid = []
	for i in range(len(t_begin)):
		difference = t_end[i] - t_begin[i]
		delta_t.append(difference)
		mid = (t_end[i] + t_begin[i])/2
		t_mid.append(mid)

	Cp_value = [57.9010205202840, 95.2180974013146, 90.5952432455687, 76.4445831298871, 63.3003938410782, 53.5705284761335, 47.0172406580489, 42.7947864442063, 40.1138049471022, 38.3948185336720, 37.2548925184664, 36.4555447590628, 35.5993410153805, 34.7500963158946, 34.0324557436423, 33.3703352992132, 32.7403259283634, 32.1349790920160, 31.5515644406990, 30.9887341399848, 30.4455548405084, 29.9212309257113, 29.4150258503855, 28.9262394471749, 24.4245775698491, 19.5132444438185, 16.7331897598208, 14.9450865710512, 13.6372965951278, 12.5787438198229]
	Cp_value = [number * 1000 for number in Cp_value]
	color = 'red'
	thickness = 3

	if plot_figures:
		plt.plot(t_mid, Cp_value, color=color, linewidth=thickness) #
		plt.grid(False)
		plt.title('FDG Values')
		plt.legend(['Input Function'])
		plt.show(block=False)

	N_indices = len(values)
	num_fits = N_indices

	list_intercept = np.zeros(N_indices)[:, np.newaxis]
	list_slope = np.zeros(N_indices)[:, np.newaxis]

	list_K = np.zeros(N_indices)[:, np.newaxis]
	list_C = np.zeros((N_frames, N_indices))

	K1_list = [0] * 7
	k2_list = [0] * 7
	k3_list = [0] * 7
	k4_list = [0] * 7
	Vp_list = [0] * 7
	i = 0
	for array in values:
		K1_list[i] = (array[0])
		k2_list[i] = (array[1])
		k3_list[i] = (array[2])
		k4_list[i] = (array[3])
		Vp_list[i] = (array[4])
		i = i + 1

	ROI_image = np.zeros((xdim, ydim, zdim))
	ROI_image_slice = np.zeros((xdim, ydim))

	ROI_image = nib.load(ROIs_filename).get_fdata()

	image_4D = np.zeros((xdim, ydim, zdim, size_t))

	intercept_image = np.zeros((xdim, ydim, zdim))
	slope_image = np.zeros((xdim, ydim, zdim))
	DV_image = np.zeros((xdim, ydim, zdim))

	intercept_image_REF = np.zeros((xdim, ydim, zdim))
	slope_image_REF = np.zeros((xdim, ydim, zdim))
	DVR_image = np.zeros((xdim, ydim, zdim))

	delt = 0.01
	t_interp = np.arange(0, t_end[N_frames - 1] + delt * 2, delt)
	f_linear = interp1d(t_mid, Cp_value, 'linear', fill_value='extrapolate')
	Cp_value_interp = f_linear(t_interp)

	cp_t = Cp_value
	cp_t2 = Cp_value_interp

	t = t_mid
	t2 = t_interp
	size_t2 = len(t2)

	cp = np.zeros(N_frames)
	sp = np.zeros(N_frames)

	for f in np.arange(0, N_frames):
		f0 = 0
		while t_interp[f0] < t_mid[f]:
			sp[f] = sp[f] + Cp_value_interp[f0] * delt
			f0 = f0 + 1
		cp[f] = Cp_value_interp[f0]

	cp_NP6 = np.zeros(N1)
	sp_NP6 = np.zeros(N1)

	t_interp_NP6 = np.arange(t_begin[N0], t_end[N_frames - 1] + delt, delt)
	Cp_value_NP6 = Cp_value[N0:N0 + N1]
	t_mid_NP6 = t_mid[N0:N0 + N1]
	f_linear2 = interp1d(t_mid_NP6, Cp_value_NP6, 'linear', fill_value='extrapolate')
	Cp_NP6_value_interp = f_linear2(t_interp_NP6)

	for f in np.arange(0, N1):
		f0 = 0
		while t_interp_NP6[f0] < t_mid_NP6[f]:
			sp_NP6[f] = sp_NP6[f] + Cp_NP6_value_interp[f0]*delt
			f0 = f0 + 1
		cp_NP6[f] = Cp_NP6_value_interp[f0]

	true_integral = sp[N0] - sp_NP6[0]

	for index in np.arange(0, N_indices):
		K1 = K1_list[index]
		k2 = k2_list[index]
		k3 = k3_list[index]
		k4 = k4_list[index]
		Vp = Vp_list[index]

		a1 = (k2 + k3 + k4 - math.sqrt((k2 + k3 + k4)**2 - 4 * k2 * k4))/2
		a2 = (k2 + k3 + k4 + math.sqrt((k2 + k3 + k4)**2 - 4 * k2 * k4))/2

		hold0 = K1/(a2-a1) * ((k3 + k4 - a1) * np.exp(-a1 * t2) + (a2 - k3 - k4) * np.exp(-a2 * t2))
		hold0_r = hold0.reshape(1, hold0.shape[0])
		cp_t2_r = cp_t2.reshape(1, cp_t2.shape[0])
		result0 = convolve2d(hold0_r, cp_t2_r) * delt

		result = result0[:, 0:size_t2] + Vp * cp_t2

		f_interp = interp1d(t2, result)
		C = f_interp(t)

		K = K1 * k3/(k2 + k3)

		list_K[index] = K
		list_C[:, index] = C

	for index in np.arange(0, N_indices):
		C = list_C[:, index]

		y0 = C/cp
		x0 = sp/cp

		y = y0[N0:N_frames]
		x = x0[N0:N_frames]
		Ones = np.ones(N1)

		y_t = y[:, np.newaxis]
		X = np.zeros((6, 2))
		for i in np.arange(0, N1):
			X[i,:] = [x[i], Ones[i]]

		model = LinearRegression()
		model.fit(X, y_t)
		b = model.coef_

		list_slope[index] = b[0][0]
		list_intercept[index] = b[0][1]

		if plot_figures:
			fig, ax = plt.subplots(1)
			ax.set_xlabel('t')
			ax.set_ylabel('C')

			ax.plot(t, C, linewidth=4, markeredgecolor='k', markersize=12)

			plot_style = plot_shape0[index] + plot_shape1[index] + plot_shape2[index]

			if 'o' in plot_style:
				ax.plot(t, C, marker='o')
			elif '+' in plot_style:
				ax.plot(t, C, marker='+')
			elif '^' in plot_style:
				ax.plot(t, C, marker='^')
			else:
				ax.plot(t, C)

			Legend = ['Myocardium','Blood','Background','Liver','Lung','Liver Tumor','Lung Tumor']
			plt.legend([Legend[index]])
			plt.show(block=False)

	K_image = np.zeros((xdim, ydim, zdim))
	B_image = np.zeros((xdim, ydim, zdim))

	flag = 0

	for xx in tqdm(np.arange(xdim)):
		for yy in np.arange(ydim):
			for zz in np.arange(zdim):
				index = int(ROI_image[xx, yy, zz])

				if index == 0:
					continue
				C = list_C[:, index-1]

				image_4D[xx, yy, zz, :] = C
				K_image[xx, yy, zz] = list_slope[index-1]
				B_image[xx, yy, zz] = list_intercept[index-1]

	mean_value = np.mean(np.mean(np.mean(np.mean(image_4D))))

	for i in np.arange(N1):
		index = i + N0
		filename = 'input_images_frame{}.nii'.format(i + 1)
		filepath = os.path.join(output_path, filename)
		finalized_input = nib.Nifti1Image(image_4D[:,:,:,index], affine=np.eye(4))
		nib.save(finalized_input, filepath)

	with open(sp_list_filename, 'w') as file:
		for value in sp_NP6:
			file.write(str(value) + '\n')

	with open(cp_list_filename, 'w') as file:
		for value in cp_NP6:
			file.write(str(value) + '\n')

	finalizedKImage = nib.Nifti1Image(K_image, affine=np.eye(4))
	KImageFilename = 'standard_fit_K_image.nii'
	KImageFilepath = os.path.join(output_path, KImageFilename)
	nib.save(finalizedKImage, KImageFilepath)

	finalizedBImage = nib.Nifti1Image(B_image, affine=np.eye(4))
	BImageFilename = 'standard_fit_B_image.nii'
	BImageFilepath = os.path.join(output_path, BImageFilename)
	nib.save(finalizedBImage, BImageFilepath)
