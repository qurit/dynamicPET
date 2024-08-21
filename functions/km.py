import os
import json
import numpy as np
import nibabel as nib
from scipy.interpolate import interp1d
from scipy import integrate
import math
from scipy.signal import convolve
import warnings
warnings.filterwarnings("ignore", category=integrate.IntegrationWarning)

def generate_graphics(output_path, config, kinetic_parameters, ROIs_filename, xdim, ydim, zdim):
    input_frame_durations = config["input_frame_durations"]
    input_frame_starts = config["input_frame_starts"]
    input_function_time = config["input_function_time"]
    input_function_concentration = config["input_function_concentration"]
    output_frame_durations = config["output_frame_durations"]
    output_frame_starts = config["output_frame_starts"]

    if not input_frame_starts:
        input_frame_starts = [0] + np.cumsum(input_frame_durations[:-1]).tolist()/60

    if not input_function_time:
        tmid_in = [(i+j/2)/60 for i,j in zip(input_frame_starts, input_frame_durations)]
    else:
        tmid_in = [i/60 for i in input_function_time]

    if not output_frame_durations:
        output_frame_durations = input_frame_durations
        output_frame_starts = input_frame_starts

    input_function = [number * 1000 for number in input_function_concentration]
    tmid_out = [(i+j/2)/60 for i,j in zip(output_frame_starts, output_frame_durations)]

    N_frames = len(output_frame_durations)
    N_regions = len(kinetic_parameters)

    ROI_image = np.zeros((xdim, ydim, zdim))
    ROI_image = nib.load(ROIs_filename).get_fdata()
    image_4D = np.memmap('image_4D.txt', shape = (xdim, ydim, zdim, N_frames), dtype=np.float32, mode='w+') #better way?????
    
    f_linear = interp1d(tmid_in, input_function, 'linear', fill_value='extrapolate')

    end_time = (input_frame_starts[-1]+input_frame_durations[-1])/60
    delt = 0.01
    t = np.arange(0, end_time + delt * 2, delt)
    cpt = f_linear(t)

    C = np.zeros((N_regions, len(t)))

    for i in np.arange(0, N_regions):

        K1, k2, k3, k4, Vp = np.array(kinetic_parameters)[i]

        a1 = (k2 + k3 + k4 - math.sqrt((k2 + k3 + k4)**2 - 4 * k2 * k4))/2
        a2 = (k2 + k3 + k4 + math.sqrt((k2 + k3 + k4)**2 - 4 * k2 * k4))/2

        temp = K1/(a2-a1) * ((k3 + k4 - a1) * np.exp(-a1 * t) + (a2 - k3 - k4) * np.exp(-a2 * t))

        C[i, :] = (1-Vp) * convolve(temp, cpt, method='direct')[0:len(t)] * delt + Vp*cpt

    C_interp = interp1d(t, C)
    sampled_C = C_interp(tmid_out)

    regions = (ROI_image.astype(int) - 1)
    image_4D[regions>=0,:] = sampled_C[regions[regions>=0], :]

    for i in range(len(tmid_out)):
        filename = 'input_images_frame{}.nii'.format(i + 1)
        filepath = os.path.join(output_path, filename)
        final_image = np.array(image_4D[:,:,:,i])
        finalized_input = nib.Nifti1Image(final_image, affine=np.eye(4))
        nib.save(finalized_input, filepath)

    kp =  ['K1', 'k2', 'k3', 'k4', 'Vp']
    image = np.zeros((xdim, ydim, zdim))

    for i, p in enumerate(kp):
        image[regions>=0] = np.array(kinetic_parameters)[regions[regions>=0],i]
        filename = 'grouund_truth_{}_image.nii'.format(p)
        filepath = os.path.join(output_path, filename)
        finalized_input = nib.Nifti1Image(image, affine=np.eye(4))
        nib.save(finalized_input, filepath)

    os.remove('image_4D.txt')
    return

if __name__ == '__main__':
    with open("config.json", 'r') as f:
        config = json.load(f)

    with open(os.path.join('input', 'kinetic_parameters.json'), 'r') as file:
        parameters = json.load(file)
    kinetic_parameters = [value for key, value in parameters.items()]
    organs = [key for key, value in parameters.items()]

    generate_graphics('output', config, kinetic_parameters, 'input/NCAT_128x128x47_0347x0347x0327_regions.nii', 128,128,47)