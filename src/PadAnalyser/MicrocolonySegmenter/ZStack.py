import cv2 as cv
import numpy as np
from typing import Optional, List
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from . import MKSegmentUtils

def laplacian(frame_raw):
    # compute laplacian compressed stack
    KERNEL_SIZE = 9
    laplacian_frame = cv.GaussianBlur(frame_raw, (KERNEL_SIZE, KERNEL_SIZE), 0) # blur, kernel size about feature size
    laplacian_frame = cv.Laplacian(laplacian_frame, cv.CV_32S, ksize=KERNEL_SIZE) # laplacian
    laplacian_frame = laplacian_frame//2**16 # scale to fit in int16
    laplacian_frame = laplacian_frame.astype(np.int16)

    return laplacian_frame

# 1. Split frame into regions.
# 2. For each region, compute preffered index. If no preferred index, return none.
# 3. Based on these indices, determine the global tilt of the FOV and make an equation for the plane.
# 4. Using the plane equation, make a matrix spesifing which pixels should be used. Linearly interpolate neighouring values to the plane, and place zero for everything else. 
# 5. Using this matrix, make a z-stack projection to the single in-focus frame. 

# Define the plane equation using the coefficients
def plane_equation(x, y, coeffs): 
    return coeffs[0] * x + coeffs[1] * y + coeffs[2]


def determine_global_tilt(center_positions, preferred_indices):
    """
    Determine the global tilt of the FOV and create an equation for the plane.
    This function fits a plane to the preferred indices using least squares method.
    """
    points = []
    values = []
    
    for center_row, preferred_indices_row in zip(center_positions, preferred_indices):
        for (x,y), index in zip(center_row, preferred_indices_row):
            if index is not None:
                points.append([x, y, 1])
                values.append(index)
    
    if not points:
        raise ValueError("No preferred indices provided")
    
    # Convert points and values to numpy arrays
    points = np.array(points)
    values = np.array(values)
    
    # Solve for the plane coefficients (a, b, c) in the equation ax + by + c = z
    coeffs, _, _, _ = np.linalg.lstsq(points, values, rcond=None)
    
    return coeffs


def create_pixel_mask(frame_shape, plane_coefficients):
    """
    Create a pixel mask based on the plane equation using piecewise linear interpolation between neighouring indices.
    The output mask has the same shape as the input frame_shape (z,x,y).
    
    Returns:
    - mask: 3D numpy array with values between 0 and 1 representing the mask values based on the linear interpolation between two neighboring z-indices.
    """
    depth, height, width = frame_shape
    np.zeros(frame_shape)

    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    
    # Create a mesh grid
    x, y = np.meshgrid(x, y)

    # Calculate the z values based on the plane equation
    z_vals = plane_equation(x, y, plane_coefficients)
    
    # clamp z values to the range [0, depth-1]
    z_vals = np.clip(z_vals, 0, depth-1)

    # Create an empty mask with the given frame shape
    mask = np.zeros(frame_shape)

    for z in range(depth):
        dist = np.abs(z - z_vals)
        mask[z, :, :] = np.where(dist < 1, 1-dist, 0)

    return mask



def find_peaks_including_boundries(values):

    peaks, _ = find_peaks(values)
    # Including boundaries in the peaks if they are maxima
    if values[0] > values[1]:
        peaks = np.insert(peaks, 0, 0)
    if values[-1] > values[-2]:
        peaks = np.append(peaks, len(values) - 1)

    return peaks


def compute_preferred_index(z_stack):
    """
    Compute the preferred index for each region.

    Parameters:
    - z_stack: 3D numpy array representing the z-stack of images (z, height, width).
    Returns:
    - preferred_indices: 2D numpy array where each element represents the preferred index for the corresponding region in the regions parameter.
    """
    # Step 1: Compute the mean intensity value for each region in each frame

    z_stack = np.minimum(z_stack, 0)

    means = np.mean(z_stack, axis=(1, 2))

    if np.max(means) - np.min(means) < 1:
        return None

    # Step 2: Find local maxima along z
    
    peaks = find_peaks_including_boundries(-means)
    
    # plt.figure()
    # plt.plot(means)
    # plt.plot(peaks, means[peaks], "x")
    
    if peaks.size:
        # Step 3: If there are multiple peaks, choose the lower z-index
        return peaks[0]

    return None



def project_to_plane(zstack: List[np.ndarray], dinfo, plane_coefficients=None):

    if plane_coefficients == None:
        ls = np.array([laplacian(f) for f in zstack])

        region_size = (256, 256)  
        # region_size = (128, 128)  

        center_positions = [[
                (y+region_size[1]//2, x+region_size[0]//2) 
                for y in range(0, ls.shape[2], region_size[1])
            ] for x in range(0, ls.shape[1], region_size[0])
        ]

        z_scores = [[
                compute_preferred_index(ls[:, x:x+region_size[0], y:y+region_size[1]])
                for y in range(0, ls.shape[2], region_size[1])
            ] for x in range(0, ls.shape[1], region_size[0])
        ]

        if dinfo.printing:
            print(np.array(z_scores))

        plane_coefficients = determine_global_tilt(center_positions, z_scores)

    mask = create_pixel_mask(ls.shape, plane_coefficients)

    selected_indices_approx = np.argmax(mask, axis=0)
    MKSegmentUtils.plot_frame(selected_indices_approx, dinfo=dinfo.append_to_label('selected_indices'))

    fs = np.array(zstack)
    f_best = np.mean(fs*mask, axis=0)
    MKSegmentUtils.plot_frame(f_best, dinfo=dinfo.append_to_label('z_best'))

    return f_best, plane_coefficients