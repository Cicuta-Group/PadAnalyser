import os
import cv2 as cv
import numpy as np
from typing import Optional, List
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from . import MKSegmentUtils

import skimage.transform
import skimage.measure


# def z_stack_projection(stack, dinfo: DInfo):
#     fs = stack
    
#     # fs = [np.maximum(f-np.mean(fs), 0) for f in fs]

#     # WINDOW_SIZE = 401  # choose an appropriate window size
    
#     # # Compute the local mean for each frame
#     # local_means = [cv.boxFilter(f.astype(float), -1, (WINDOW_SIZE, WINDOW_SIZE)) for f in fs]
#     # fs = [np.maximum(f - mean, 0) for f, mean in zip(fs, local_means)]
    
#     # for i, frame in enumerate(fs):
#     #     print(f'frame {i} max {np.max(frame)}')
#     #     MKSegmentUtils.plot_frame(frame, dinfo=dinfo.append_to_label(f'rbg {i}'))

#     # # convert to uint8
#     # fs = [f.astype(np.uint8) for f in fs]

#     # Find second order gradients of each frame
#     fs = [cv.GaussianBlur(f, (5, 5), 0) for f in fs] # blur, kernel size about feature size
#     fs = [cv.Laplacian(f, cv.CV_32S, ksize=7) for f in fs] # laplacian

#     # Only keep negative gradients and make them positive (corresponds to area inside cells when in focus)
#     fs = [np.maximum(-f, 0) for f in fs]
    
#     # Compute focus score for each pixel by downsampling with funciton that characterize information. Varience best when testing.
#     KERNEL_SIZE = 61
#     fs = [skimage.transform.resize(skimage.measure.block_reduce(f, (KERNEL_SIZE, KERNEL_SIZE), np.var), (f.shape)) for f in fs]
    
#     # Find index with highest score in stack for each pixel and pick in focus frame based on that
#     f_max = np.argmax(np.array(fs), 0)
#     f_focus = np.take_along_axis(np.array(stack), f_max[None, ...], axis=0)[0]

#     # print('new')
#     # # Compute local variance and mean
#     # KERNEL_SIZE = 101
#     # local_var = [cv.boxFilter(f**2,  -1, (KERNEL_SIZE, KERNEL_SIZE)) - cv.boxFilter(f, -1, (KERNEL_SIZE, KERNEL_SIZE))**2 for f in fs]
#     # local_mean = [cv.boxFilter(f, -1, (KERNEL_SIZE, KERNEL_SIZE)) / (KERNEL_SIZE**2) for f in fs]

#     # epsilon = 1e-5  # To prevent division by zero
#     # focus_scores = [variance / (mean + epsilon) for variance, mean in zip(local_var, local_mean)]

#     # # Find index with highest score in stack for each pixel and pick in-focus frame based on that
#     # f_max = np.argmax(np.array(focus_scores), 0)
#     # f_focus = np.take_along_axis(np.array(stack), f_max[None, ...], axis=0)[0]

#     # # Check histogram and invert if needed
#     # hist, bins = np.histogram(f_focus, bins=256, range=(0, 256))
#     # peak = bins[np.argmax(hist)]
#     # if peak < 128:  # Assuming 8-bit images with values between 0 and 255
#     #     f_focus = 255 - f_focus

#     MKSegmentUtils.plot_frame(f_max, dinfo=dinfo.append_to_label('z_stack_indices'))
#     MKSegmentUtils.plot_frame(f_focus, dinfo=dinfo.append_to_label('z_stack_best'))

#     return f_focus





def laplacian(frame_raw):
    # compute laplacian compressed stack
    KERNEL_SIZE = 9
    laplacian_frame = cv.GaussianBlur(frame_raw, (KERNEL_SIZE, KERNEL_SIZE), 0) # blur, kernel size about feature size
    laplacian_frame = cv.Laplacian(laplacian_frame, cv.CV_32S, ksize=KERNEL_SIZE) # laplacian
    laplacian_frame = laplacian_frame//2**16 # scale to fit in int16
    laplacian_frame = laplacian_frame.astype(np.int16)

    return laplacian_frame



# Contounous projection based on low pass filter of laplacian areas. 
def z_stack_projection_laplace(stack, dinfo):
    fs = stack

    # Compute gradients
    fs = [laplacian(f) for f in fs]
    
    # Only keep negative gradients (center of cells)
    fs = [np.maximum(-f, 0) for f in fs]
    
    # Compute focus score for each pixel by downsampling
    KERNEL_SIZE = 101
    fs = [skimage.transform.resize(skimage.measure.block_reduce(f, (KERNEL_SIZE, KERNEL_SIZE), np.mean), (f.shape)) for f in fs]

    # Find index with highest score in stack for each pixel
    f_max = np.argmax(np.array(fs), 0)
    f_focus = np.take_along_axis(np.array(stack), f_max[None, ...], axis=0)[0]
    
    # Check histogram and invert if needed
    # hist, bins = np.histogram(f_focus, bins=256, range=(0, 256))
    # peak = bins[np.argmax(hist)]
    # if peak < 128:  # Assuming 8-bit images with values between 0 and 255
    #     f_focus = 255 - f_focus
    
    MKSegmentUtils.plot_frame(f_max, dinfo=dinfo.append_to_label('z_stack_indices'))
    MKSegmentUtils.plot_frame(f_focus, dinfo=dinfo.append_to_label('z_stack_best'))

    return f_focus


# Contounous projection based on low pass filter of laplacian areas. 
def z_stack_projection_sobel(stack, dinfo):
    fs = stack

    # Apply Gaussian blur
    fs_float = [cv.GaussianBlur(f, (5, 5), 0).astype(np.float32) for f in fs]

    # Compute gradients using Sobel
    gradients = [cv.Sobel(f, cv.CV_32F, 1, 1, ksize=3) for f in fs_float]
    fs = gradients # [np.sqrt(np.square(grad[:,:,0]) + np.square(grad[:,:,1])) for grad in gradients]
    
    # Only keep positive gradients
    fs = [np.maximum(f, 0) for f in fs]
    
    # Compute focus score for each pixel by downsampling
    KERNEL_SIZE = 101
    fs = [skimage.transform.resize(skimage.measure.block_reduce(f, (KERNEL_SIZE, KERNEL_SIZE), np.mean), (f.shape)) for f in fs]

    # Find index with highest score in stack for each pixel
    f_max = np.argmax(np.array(fs), 0)
    f_focus = np.take_along_axis(np.array(stack), f_max[None, ...], axis=0)[0]
    
    # Check histogram and invert if needed
    # hist, bins = np.histogram(f_focus, bins=256, range=(0, 256))
    # peak = bins[np.argmax(hist)]
    # if peak < 128:  # Assuming 8-bit images with values between 0 and 255
    #     f_focus = 255 - f_focus
    
    MKSegmentUtils.plot_frame(f_max, dinfo=dinfo.append_to_label('z_stack_indices'))
    MKSegmentUtils.plot_frame(f_focus, dinfo=dinfo.append_to_label('z_stack_best_'))

    return f_focus



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
    if len(points) < 3:
        return (0, 0, np.mean(values)) # If only one or two points, return a flat plane

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


MIN_PROMINANE = 0.4
def find_peaks_including_boundries(values):

    peaks, _ = find_peaks(values, prominence=MIN_PROMINANE)
    # Including boundaries in the peaks if they are maxima
    if values[0] > values[1] + MIN_PROMINANE:
        peaks = np.insert(peaks, 0, 0)
    if values[-1] > values[-2] + MIN_PROMINANE:
        peaks = np.append(peaks, len(values) - 1)

    return peaks


def compute_preferred_index(z_stack, dinfo):
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
        return None, means, None

    # Step 2: Find local maxima along z
    
    peaks = find_peaks_including_boundries(-means)
    
    # if dinfo.live_plot:
    #     plt.figure()
    #     plt.plot(means)
    #     plt.plot(peaks[0], means[peaks][0], "0")
    #     plt.plot(peaks[1:], means[peaks][1:], "x")
    
    if peaks.size:
        # Step 3: If there are multiple peaks, choose the lower z-index
        return peaks[0], means, peaks

    return None, means, None


def plot_z_scores(z_scores, z_score_means, z_score_peaks, dinfo):
    
    if dinfo.live_plot or dinfo.file_plot:
        
        x_len, y_len = len(z_scores), len(z_scores[0])
        
        # all_means = [item for sublist in z_score_means for item in sublist]
        y_min = np.nanmin(z_score_means)
        y_max = np.nanmax(z_score_means)

        fig = plt.figure(figsize=(6,4), dpi=300)
        gs = fig.add_gridspec(x_len, y_len, hspace=0, wspace=0)
        axs = gs.subplots(sharex='col', sharey='row')
        fig.suptitle('y-range = [{:.2f}, {:.2f}]'.format(y_min, y_max), fontsize=16)
        for axs_row, z_score_means_row, z_score_peak_row in zip(axs, z_score_means, z_score_peaks):
            for ax, z_score_mean, z_score_peak in zip(axs_row, z_score_means_row, z_score_peak_row):
                ax.plot(z_score_mean)
                if not isinstance(z_score_peak, type(None)):
                    ax.plot(z_score_peak[0], z_score_mean[z_score_peak][0], "x")
                    ax.plot(z_score_peak[1:], z_score_mean[z_score_peak][1:], ".")
                ax.set_ylim(y_min, y_max)
                ax.set_xlim(0, len(z_score_means[0]))
                ax.set_axis_off()
        
        if dinfo.file_plot:
            filepath = os.path.join(dinfo.image_dir, dinfo.label + '_z_scores.svg')
            plt.savefig(filepath, bbox_inches='tight')



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

        z_scores_data = [[
                compute_preferred_index(ls[:, x:x+region_size[0], y:y+region_size[1]], dinfo=dinfo.append_to_label(f'z_{x}_{y}'))
                for y in range(0, ls.shape[2], region_size[1])
            ] for x in range(0, ls.shape[1], region_size[0])
        ]

        z_scores, z_score_means, z_score_peaks = zip(*[[list(z) for z in zip(*row)] for row in z_scores_data])
        plot_z_scores(z_scores, z_score_means, z_score_peaks, dinfo)

        if dinfo.printing:
            print(np.array(z_scores))

        try:
            plane_coefficients = determine_global_tilt(center_positions, z_scores)
        except ValueError:
            return z_stack_projection_laplace(zstack, dinfo=dinfo), None

    mask = create_pixel_mask(ls.shape, plane_coefficients)

    selected_indices_approx = np.argmax(mask, axis=0)
    MKSegmentUtils.plot_frame(selected_indices_approx, dinfo=dinfo.append_to_label('selected_indices'))

    fs = np.array(zstack)
    f_best = np.mean(fs*mask, axis=0)

    MKSegmentUtils.plot_frame(f_best, dinfo=dinfo.append_to_label('f_best'))

    return f_best, plane_coefficients