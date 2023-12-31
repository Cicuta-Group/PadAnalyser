import os
import cv2 as cv
import numpy as np
from typing import Optional, List
import matplotlib.pyplot as plt
from scipy.signal import find_peaks 

from . import MKSegmentUtils

import skimage.transform
import skimage.measure
import logging

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



import scipy.signal


WINDOW_SIZE = 200 # 200 is close to GCD of 3208x2200, 240 is gcd of pixel counts (1920x1200)
BLUR_SIZE = 51

def window_kernel(xs, ys, shape):
    k = np.zeros(shape).astype(np.uint16)
    k[ys,xs] = 255
    # k = cv.GaussianBlur(k, (51, 51), 0)
    k = cv.blur(k, (BLUR_SIZE, BLUR_SIZE))
    k = k.astype(np.float32)/255.0
    # plot_frame(k.astype(np.uint16), 'kernel', plot=xs.start == 0 and ys.start == 0)
    return k


WINDOW_SIZE_TILES = 200
score_kernel = np.array([ # put weight on neigbouring tiles as well, to improve continuity
    [1,1,1],
    [1,2,1],
    [1,1,1],
]) 

# Better method when backlash for stage motion in z-stack is too large, making in-focus peaks too wide.
def project_with_tiles(stack, dinfo):
    
    fs = stack
    fs = [cv.GaussianBlur(f, (7, 7), 0) for f in fs] # blur, kernel size about feature size
    fs = [cv.Laplacian(f, cv.CV_32S, ksize=7) for f in fs] # laplacian
    fs_abs = [np.square(f) for f in fs] # square to make all positive and emphasize large values over larger area with smaller amplitude

    # size-representative frame
    height, width = stack[0].shape
    x_splits = list(range(0, width+1, int(WINDOW_SIZE_TILES)))
    y_splits = list(range(0, height+1, int(WINDOW_SIZE_TILES)))

    x_ranges = [slice(a,b) for a,b in zip(x_splits, x_splits[1:])]
    y_ranges = [slice(a,b) for a,b in zip(y_splits, y_splits[1:])]

    weighted_scores = [np.array([[np.mean(l[ys,xs]) for ys in y_ranges] for xs in x_ranges]) for l in fs_abs]
    weighted_scores = [scipy.signal.convolve2d(d, score_kernel, mode='same') for d in weighted_scores]

    f = np.zeros_like(stack[0]).astype(np.float32) # new frame to build and return
    for i, xs in enumerate(x_ranges):
        for j, ys in enumerate(y_ranges):
            scores = [l[i,j] for l in weighted_scores]
            focus_index = scores.index(max(scores))
            
            f = f + window_kernel(xs, ys, f.shape) * stack[focus_index] # add this window to return frame

    return f.astype(stack[0].dtype)


# '''
# Contounous projection based on low pass filter of laplacian areas. 
# Simplest approach, and used for initial datasets. 
# '''
# def z_stack_projection_laplace(stack, dinfo):
#     fs = stack

#     # Compute gradients
#     fs = [laplacian(f) for f in fs]
    
#     # Only keep negative gradients (center of cells)
#     fs = [np.maximum(-f, 0) for f in fs]
    
#     # Compute focus score for each pixel by downsampling
#     KERNEL_SIZE = 101
#     fs = [skimage.transform.resize(skimage.measure.block_reduce(f, (KERNEL_SIZE, KERNEL_SIZE), np.mean), (f.shape)) for f in fs]

#     # Find index with highest score in stack for each pixel
#     f_max = np.argmax(np.array(fs), 0)
#     f_focus = np.take_along_axis(np.array(stack), f_max[None, ...], axis=0)[0]
    
#     # Check histogram and invert if needed
#     # hist, bins = np.histogram(f_focus, bins=256, range=(0, 256))
#     # peak = bins[np.argmax(hist)]
#     # if peak < 128:  # Assuming 8-bit images with values between 0 and 255
#     #     f_focus = 255 - f_focus
    
#     MKSegmentUtils.plot_frame(f_max, dinfo=dinfo.append_to_label('z_stack_indices'))
#     MKSegmentUtils.plot_frame(f_focus, dinfo=dinfo.append_to_label('z_stack_best'))

#     return f_focus


# # Contounous projection based on low pass filter of laplacian areas. 
# def z_stack_projection_sobel(stack, dinfo):
#     fs = stack

#     # Apply Gaussian blur
#     fs_float = [cv.GaussianBlur(f, (5, 5), 0).astype(np.float32) for f in fs]

#     # Compute gradients using Sobel
#     gradients = [cv.Sobel(f, cv.CV_32F, 1, 1, ksize=3) for f in fs_float]
#     fs = gradients # [np.sqrt(np.square(grad[:,:,0]) + np.square(grad[:,:,1])) for grad in gradients]
    
#     # Only keep positive gradients
#     fs = [np.maximum(f, 0) for f in fs]
    
#     # Compute focus score for each pixel by downsampling
#     KERNEL_SIZE = 101
#     fs = [skimage.transform.resize(skimage.measure.block_reduce(f, (KERNEL_SIZE, KERNEL_SIZE), np.mean), (f.shape)) for f in fs]

#     # Find index with highest score in stack for each pixel
#     f_max = np.argmax(np.array(fs), 0)
#     f_focus = np.take_along_axis(np.array(stack), f_max[None, ...], axis=0)[0]
    
#     # Check histogram and invert if needed
#     # hist, bins = np.histogram(f_focus, bins=256, range=(0, 256))
#     # peak = bins[np.argmax(hist)]
#     # if peak < 128:  # Assuming 8-bit images with values between 0 and 255
#     #     f_focus = 255 - f_focus
    
#     MKSegmentUtils.plot_frame(f_max, dinfo=dinfo.append_to_label('z_stack_indices'))
#     MKSegmentUtils.plot_frame(f_focus, dinfo=dinfo.append_to_label('z_stack_best_'))

#     return f_focus





### New approach  - fit plane

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


MIN_PROMINENCE = 0.2

def find_peaks_including_boundries(values):

    peaks, _ = find_peaks(values, prominence=MIN_PROMINENCE)
    
    # Including boundaries in the peaks if they are maxima
    if values[0] > values[1] + MIN_PROMINENCE:
        peaks = np.insert(peaks, 0, 0)
    if values[-1] > values[-2] + MIN_PROMINENCE:
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

    if np.max(means) - np.min(means) < MIN_PROMINENCE:
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

    if zstack[0].dtype == np.uint8:
        if dinfo.printing:
            print('Warning: zstack is uint8, converting to uint16')
        zstack = [f.astype(np.uint16)*255 for f in zstack]

    if dinfo.printing:
        if zstack[0].dtype != np.uint16:
            print('Warning: zstack is not uint16')
         
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
            return project_with_tiles(zstack, dinfo=dinfo), None # fallback 

    mask = create_pixel_mask(ls.shape, plane_coefficients)

    selected_indices_approx = np.argmax(mask, axis=0)
    MKSegmentUtils.plot_frame(selected_indices_approx, dinfo=dinfo.append_to_label('selected_indices'))

    fs = np.array(zstack)
    f_best = np.mean(fs*mask, axis=0)

    MKSegmentUtils.plot_frame(f_best, dinfo=dinfo.append_to_label('f_best'))

    return f_best, plane_coefficients


def clip(frame, percentile=99.999): # truncate about 64 pixels for 99.999 percentile
    try:
        lower_percentile, upper_percentile = np.percentile(frame, q=[100-percentile, percentile])    
    except Exception as e:
        logging.error(f'percentile calculation exception {e}, {percentile}, {frame}')
        print(f'percentile calculation exception {e}, {percentile}, {frame}')
        raise Exception(f'percentile calculation exception {e}, {percentile}, {frame}')
        
    return np.clip(frame, lower_percentile, upper_percentile)


'''
Master method that takes a stack of frames and returns a single in-focus frame.
Tries to find best-fit plane, and if it fails, falls back to laplacian projection.
'''
def flatten_stack(stack, dinfo, large_backlash=False):

    plane_coefficients = None
    
    # Check if stack is already flattened, otherwise compute projection
    if isinstance(stack, np.ndarray): frame_raw = stack
    elif len(stack) == 1: frame_raw = stack[0]
    elif large_backlash: frame_raw, plane_coefficients = project_with_tiles(stack, dinfo=dinfo), None # compute laplacian from normalized frame
    else: frame_raw, plane_coefficients = project_to_plane(stack, dinfo=dinfo) # compute laplacian from normalized frame

    if frame_raw.dtype == np.uint8: frame_raw = frame_raw.astype(np.uint16)*255
    
    frame_clipped = clip(frame_raw)
    frame_blurred = cv.GaussianBlur(frame_clipped, (3, 3), 0) # blur before norm to reduce outliers
    frame8 = MKSegmentUtils.norm(frame_blurred)
    
    # frame16 = MKSegmentUtils.normanlize_uint16(frame_raw)
    # frame = MKSegmentUtils.to_dtype_uint8(frame_raw)

    # # compute laplacian compressed stack
    # laplacian_frame = cv.GaussianBlur(frame16, (7, 7), 0) # blur, kernel size about feature size
    # laplacian_frame = cv.Laplacian(laplacian_frame, cv.CV_32S, ksize=7) # laplacian
    # laplacian_frame = laplacian_frame//2**16 # scale to fit in int16
    # laplacian_frame = laplacian_frame.astype(np.int16)
    
    # output debug frames
    # MKSegmentUtils.plot_frame(frame16, dinfo=dinfo.append_to_label('z_stack_frame16'))
    MKSegmentUtils.plot_frame(frame8, dinfo=dinfo.append_to_label('z_stack_frame8'))
    # MKSegmentUtils.plot_frame(laplacian_frame, dinfo=dinfo.append_to_label(f'z_stack_laplacian'))
    # for i, s in enumerate(stack):
    #     MKSegmentUtils.plot_frame(s, dinfo=dinfo.append_to_label(f'z_stack_frame_{i}'))

    return frame8, plane_coefficients