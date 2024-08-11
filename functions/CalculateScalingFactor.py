import numpy as np


def scaling_factor(BqDim_ind, d_x, d_y, d_z, t1, t2, lambda_val, calibration_factor, IMAGE_DECAYED, TOF_factor):
    # Bq unit correction
    if BqDim_ind == 1:
        ratio = 1
    elif BqDim_ind == 2:
        ratio = 1e3
    elif BqDim_ind == 3:
        ratio = 1e6
    else:
        raise ValueError("Invalid value for BqDim_ind. Should be 1, 2, or 3.")

    vol = d_x * d_y * d_z

    # Time calculation
    delta_t = t2 - t1  # unit: [min]

    # Decay correction
    if not IMAGE_DECAYED:
        decay_factor = np.exp(-lambda_val * t1) * (1 / (lambda_val * delta_t)) * (1 - np.exp(-lambda_val * delta_t))
    else:
        decay_factor = 1

    # Scaling factor calculation
    scale_factor = ratio * vol * (60 * delta_t) * decay_factor * calibration_factor * TOF_factor

    return scale_factor
