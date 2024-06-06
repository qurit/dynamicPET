from math import ceil
import cv2
import numpy as np


def gen_nema(ImDim, VoxSize, ROIs):
    # Initialization
    ROIs[1:] = [roi - ROIs[0] for roi in ROIs[1:]]

    # Draw Upper semi-circle
    C1 = (ImDim // 2, (ImDim // 2 + round(35 / VoxSize)))
    C1c = round(147 / VoxSize)
    J1 = cv2.circle(np.ones((ImDim, ImDim)), C1, C1c, (0, 0, 0), -1)
    cv2.imshow('Circle', J1)

    R1 = (1, (ImDim // 2 + round(35 / VoxSize)) + 1, ImDim - 1, ImDim // 2)
    J2 = cv2.rectangle(J1, (R1[0], R1[1]), (R1[2], R1[3]), (255, 255, 255), -1)
    cv2.imshow('Circle', J1)
    cv2.imshow('Rectangle', J2)

    P1 = J2.copy()
    P1[J2 == J2[(ImDim // 2 + round(35 / VoxSize)) + 2, (ImDim // 2 + round(35 / VoxSize)) + 2]] = J2[1, 1]
    P1[P1 != 1] = 0
    P1 = ~P1

    # Draw lower two circles and rectangle
    C2 = np.array([(ImDim // 2 - round(70 / VoxSize), ImDim // 2 + round(35 / VoxSize), round(77 / VoxSize)),
                   (ImDim // 2 + round(70 / VoxSize), ImDim // 2 + round(35 / VoxSize), round(77 / VoxSize))])
    J = cv2.circle(np.ones((ImDim, ImDim)), (C2[0, 0], C2[0, 1]), C2[0, 2], (0, 0, 0), -1)
    J = cv2.circle(J, (C2[1, 0], C2[1, 1]), C2[1, 2], (0, 0, 0), -1)

    R2 = (ImDim // 2 - round(70 / VoxSize), ImDim // 2 + round(35 / VoxSize), ceil(140 / VoxSize), ceil(77 / VoxSize))
    J3 = cv2.rectangle(J, (R2[0], R2[1]), (R2[2], R2[3]), (0, 0, 0), -1)

    P2 = J3.copy()
    P2[J3 != 1] = 0
    P2 = ~P2
    BK = np.logical_or(P2, P1)

    # Draw six circles
    C2 = (ImDim // 2 + round(114.4 / 2 / VoxSize), ImDim // 2, ceil(37 / 2 / VoxSize))
    J_1 = cv2.circle(np.ones((ImDim, ImDim)), C2, 10, (0, 0, 0), -1)
    J_1[J_1 != 1] = 0
    J_1 = ~J_1

    C2 = (ImDim // 2 - round(114.4 / 2 / VoxSize), ImDim // 2, ceil(17 / 2 / VoxSize))
    J_4 = cv2.circle(np.ones((ImDim, ImDim)), C2, 10, (0, 0, 0), -1)
    J_4[J_4 != 1] = 0
    J_4 = ~J_4

    C2 = (ImDim // 2 + round((114.4 / 2 / VoxSize) * np.cos(60 * np.pi / 180)),
          ImDim // 2 - round((114.4 / 2 / VoxSize) * np.sin(-60 * np.pi / 180)), ceil(28 / 2 / VoxSize))
    J_2 = cv2.circle(np.ones((ImDim, ImDim)), C2, 10, (0, 0, 0), -1)
    J_2[J_2 != 1] = 0
    J_2 = ~J_2

    C2 = (ImDim // 2 - round((114.4 / 2 / VoxSize) * np.cos(60 * np.pi / 180)),
          ImDim // 2 + round((114.4 / 2 / VoxSize) * np.sin(60 * np.pi / 180)), ceil(22 / 2 / VoxSize))
    J_3 = cv2.circle(np.ones((ImDim, ImDim)), C2, 10, (0, 0, 0), -1)
    J_3[J_3 != 1] = 0
    J_3 = ~J_3

    C2 = (ImDim // 2 - round((114.4 / 2 / VoxSize) * np.cos(60 * np.pi / 180)),
          ImDim // 2 - round((114.4 / 2 / VoxSize) * np.sin(60 * np.pi / 180)), ceil(13 / 2 / VoxSize))
    J_5 = cv2.circle(np.ones((ImDim, ImDim)), C2, 10, (0, 0, 0), -1)
    J_5[J_5 != 1] = 0
    J_5 = ~J_5

    C3 = (ImDim // 2 + round((114.4 / 2 / VoxSize) * np.cos(60 * np.pi / 180)),
          ImDim // 2 - round((114.4 / 2 / VoxSize) * np.sin(60 * np.pi / 180)), ceil(10 / 2 / VoxSize))
    if C2[-1] <= C3[-1]:
        C3 = (ImDim // 2 + round((114.4 / 2 / VoxSize) * np.cos(60 * np.pi / 180)) - 1,
              ImDim // 2 - round((114.4 / 2 / VoxSize) * np.sin(60 * np.pi / 180)) - 1,
              -1 + 2 * ceil(10.5 / 2 / VoxSize), -1 + 2 * ceil(10.5 / 2 / VoxSize))
        J_6 = cv2.rectangle(np.ones((ImDim, ImDim)), (C3[0], C3[1]), (C3[2], C3[3]), (0, 0, 0), -1)
    else:
        J_6 = cv2.circle(np.ones((ImDim, ImDim)), C3, 10, (0, 0, 0), -1)

    J_6[J_6 != 1] = 0
    J_6 = ~J_6

    # Assigning values to each ROI
    P = ROIs[0] * BK + ROIs[1] * J_1 + ROIs[2] * J_2 + ROIs[3] * J_3 + ROIs[4] * J_4 + ROIs[5] * J_5 + ROIs[6] * J_6

    return P.astype(int)


# Example usage
ImDim1 = 128
VoxSize1 = 3.47
ROIs1 = [1, 2, 3, 4, 5, 6, 7]  # Replace with desired ROI values
Z = gen_nema(ImDim1, VoxSize1, ROIs1)
cv2.imshow("NEMA Phantom", Z)
cv2.waitKey(0)
cv2.destroyAllWindows()
