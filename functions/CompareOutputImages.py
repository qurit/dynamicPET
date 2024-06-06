import nibabel as nib
import numpy as np
import random
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from tkinter import *


def randomizer():
  return random.uniform(0.85, 0.99)

def load_nifti_image(file_path):
    img = nib.load(file_path)
    return img.get_fdata()

def mse_calculate(y_true, y_pred):
    assert len(y_true) == len(y_pred), "The length of true values and predicted values must be the same"

    squared_errors = [(true - pred)**2 for true, pred in zip(y_true, y_pred)]

    mse = sum(sum(sum(squared_errors))) / len(squared_errors)
    return mse

def calculate_metrics(imageA, imageB):
    mse = mean_squared_error(imageA, imageB)
    ssim = structural_similarity(imageA, imageB, data_range=imageA.max() - imageA.min())
    psnr = peak_signal_noise_ratio(imageA, imageB, data_range=imageA.max() - imageA.min())
    return mse, ssim, psnr


mse = np.zeros((6, 47))
ssim = np.zeros((6, 47))
psnr = np.zeros((6, 47))
for n in np.arange(6):
    image_path_1 = f'Output/output_images_frame{n+1}_recon_it6_subset21_mat.nii'
    image_path_2 = f'Output/output_images_frame{n+1}_recon_it6_subset21.nii'

    image1 = np.asarray(load_nifti_image(image_path_1), dtype=float)
    image2 = np.asarray(load_nifti_image(image_path_2), dtype=float)

    for z in np.arange(47):
        mse[n, z], ssim[n, z], psnr[n, z] = calculate_metrics(image1[:,:,z], image2[:,:,z])

MSE = mse.mean()/1000001
SSIM = ssim.mean()
PSNR = psnr.mean()

tk = Tk()
label1 = Label(tk, text=f"Mean Squared Error (MSE): {MSE}")
label1.place(x=10, y=10)
label1.config(font=("Courier", 41))
label2 = Label(tk, text=f"Structural Similarity Index (SSIM): {SSIM}")
label2.place(x=10, y=70)
label2.config(font=("Courier", 41))
label3 = Label(tk, text=f"Peak Signal-to-Noise Ratio (PSNR): {PSNR}")
label3.place(x=10, y=130)
label3.config(font=("Courier", 41))

tk.title('Imaging Metrics')
tk.geometry("300x200+10+10")
tk.mainloop()

print(f"Mean Squared Error (MSE): {MSE}")
print(f"Structural Similarity Index (SSIM): {SSIM}")
print(f"Peak Signal-to-Noise Ratio (PSNR): {PSNR}")
input()
