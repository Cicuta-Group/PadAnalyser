import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import logging
from functools import cache

from PIL import Image, ImageFont, ImageDraw
import os
from colorhash import ColorHash
import scipy, scipy.signal
try:
    import polylabel_pyo3
except ImportError:
    print('Could not import polylabel_pyo3')

from scipy import ndimage


UM_PER_PIXEL = 0.112 # for Genicam with 40x objective
# UM_PER_PIXEL = 0.147 # for Grashopper with 40x objective

MIN_COLONY_AREA = 2 # in um^2
MIN_CELL_AREA = 0.25 # in um^2
MIN_POINT_SEPARATION = 4 # in um

'''
Normalize frame between 0 and 255
'''
def normanlize_uint16(f):
    if f.dtype == np.bool8:
        f = np.uint8(f)
    return cv.normalize(f, None, alpha=0, beta=65535, norm_type=cv.NORM_MINMAX).astype(np.uint16)

def norm(f):
    if f.dtype == np.bool8:
        f = np.uint8(f)
    return cv.normalize(f, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX).astype(np.uint8)
    
    # if dtype == np.uint16:
    #     f = cv.normalize(f, None, alpha=0, beta=65535, norm_type=cv.NORM_MINMAX)
    #     return (f / 256).astype(np.uint8)
    # if dtype == np.uint8:
    #     return cv.normalize(f, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    

    # raise TypeError(f'norm() does not know how to handle data of type {dtype}')

'''
Scale max pixel value up to 255, but keep lower bound the same
'''
def normalize_up(f):
    maximum = np.max(f)
    max_possible = np.iinfo(f.dtype).max
    return (f.astype(np.float32)*max_possible/maximum).astype(f.dtype)

# def normalize_up(f0):
#     type_max = np.iinfo(f0.dtype).max
#     mean, type_mean = np.mean(f0), type_max/2
#     scaling_ratio = type_mean / mean
#     f = f0.astype(np.float32)*scaling_ratio
#     print(np.max(f), type_max)
#     f = np.maximum(f, type_max)
#     print(np.max(f))
#     return f.astype(f0.dtype)

def to_dtype_uint8(f):
    if f.dtype == np.uint8: return f
    if f.dtype == np.uint16: return (f//2**8).astype(np.uint8)
    if f.dtype == np.uint32: return (f//2**24).astype(np.uint8)
    if f.dtype == np.uint64: return (f//2**56).astype(np.uint8)
    if f.dtype == np.int8: return ((f//2)+2**7).astype(np.uint8)
    if f.dtype == np.int16: return (((f//2)+2**15)//2**8).astype(np.uint8)
    if f.dtype == np.int32: return (((f//2)+2**31)//2**24).astype(np.uint8)
    if f.dtype == np.int64: return (((f//2)+2**63)//2**56).astype(np.uint8)
    if f.dtype == bool: return (f.astype(np.uint8)*255).astype(np.uint8)
    if f.dtype == np.float16: return norm(f)
    if f.dtype == np.float32: return norm(f)
    if f.dtype == np.float64: return norm(f)
    raise TypeError(f'to_dtype_uint8() does not know how to handle data of type {f.dtype}')

# '''
# Scale max pixel value up to 255, but keep lower bound the same
# '''
# def norm_up(f):
#     minimum = np.min(f)
#     if f.dtype == np.uint16:
#         minimum = minimum // 256

#     if f.dtype == np.bool8:
#         f = np.uint8(f)
#     return cv.normalize(f, None, alpha=minimum, beta=255, norm_type=cv.NORM_MINMAX).astype(np.uint8)




##### Kernels ####


kL1 = np.ones((11,11), dtype=np.uint8)
kL2 = np.ones((7,7), dtype=np.uint8)
k4_square = np.array([
    [1, 1, 1, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 1, 1, 1],
])
k2_square = np.array([
    [1, 1],
    [1, 1],
]).astype(np.uint8)
k3_circle = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0],
]).astype(np.uint8)
k5_circle = np.array([
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
]).astype(np.uint8)
k9_circle = np.array([
    [0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0],
]).astype(np.uint8)
k6_square = np.array([
    [1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1],
])
k5 = np.ones((5,5))

kLC_size = 41
kLC = cv.getStructuringElement(cv.MORPH_ELLIPSE,(kLC_size,kLC_size)) # large circular kernel



### Contour manipulations

def dist_mask(x, k=0):
    return np.triu(x, k) - np.triu(x, x.shape[0]-k) > 0

def mask_indices(N,k):
    return np.mask_indices(N, mask_func=dist_mask, k=k)

def split_at_indices(contour, i0, i1):

    start_index, end_index = sorted([i0, i1])

    # Split the contour into two parts
    contour_part1 = np.concatenate([contour[:start_index], contour[end_index+1:]])
    contour_part2 = contour[start_index+1:end_index]
    return contour_part1, contour_part2

    # c_a = np.concatenate((contour[:i0], contour[i1:]))
    # c_b = contour[i0:i1]
    # return c_a, c_b


def split_at_indices_symetric(contour, i0, i1, separation_ratio=0.2):
    # Find the midpoint
    midpoint = (contour[i0] + contour[i1]) / 2.0
    
    # Find the direction vector from i0 to i1
    direction = contour[i1] - contour[i0]
    direction_norm = np.linalg.norm(direction)
    
    if direction_norm == 0:  # Prevent division by zero
        return contour, []

    unit_direction = direction / direction_norm

    # Offset the indices
    separation_distance = separation_ratio * direction_norm
    offset_i0 = midpoint - (separation_distance / 2.0) * unit_direction
    offset_i1 = midpoint + (separation_distance / 2.0) * unit_direction
    
    # Create the split contours
    c_a = np.concatenate((contour[:i0], [offset_i0], [offset_i1], contour[i1:]))
    c_b = np.concatenate(([offset_i0], contour[i0:i1], [offset_i1]))

    return c_a, c_b


from scipy.spatial import cKDTree

def points_within_radius(point, points_list, radius):
    tree = cKDTree(points_list)
    indices = tree.query_ball_point(point, radius)
    return indices


from scipy import spatial

# def split_contour_by_point_distance(contour: np.array, min_distance: float = 2, preview=False):
    
#     contour_reduced = contour[:,0,:]
#     N = len(contour)
#     k = 10
#     rows, cols = mask_indices(N,k) # offset by k (only look at points separated by k or more points)
    
#     print(rows.shape, cols.shape)

#     if rows.shape[0]:
        
#         d_matrix = spatial.distance_matrix(x=contour_reduced, y=contour_reduced)
#         # print(d_matrix, rows, cols)

#         ci = np.argmin(d_matrix[rows,cols]) 
#         row, col = rows[ci], cols[ci]

#         if d_matrix[row, col] <= min_distance:
#             ca, cb = split_at_indices(contour=contour, i0=row, i1=col) # split contour on index set (row, col)

#             # check areas are sufficiently large
#             # if contour_to_area(ca) < MIN_CELL_AREA and contour_to_area(cb) < MIN_CELL_AREA: 

#             if preview:
#                 plt.figure()
#                 plt.title('Point distance')
#                 plt.plot(contour[:,0,0], contour[:,0,1], '--')
#                 plt.fill(ca[:,0,0], ca[:,0,1], '-o')
#                 plt.fill(cb[:,0,0], cb[:,0,1], '-o')
#                 plt.plot(contour[row,0,0], contour[row,0,1], 'co')
#                 plt.plot(contour[col,0,0], contour[col,0,1], 'co')
#                 plt.axis('equal')

#             return split_contour_by_point_distance(ca, min_distance=min_distance, preview=preview) + split_contour_by_point_distance(cb, min_distance=min_distance, preview=preview)

#     return [contour]





'''
Calcualtes curvature between all consecutive points in x,y arrays. 
Assumes shape is closed. 
Returns 0 when three points form straight line, negative when curving outword and positive when curving inword.
'''
def curvature(contour):

    x,y = contour[:,0,0], contour[:,0,1]
    dx = np.diff(x, prepend=x[-1])
    dy = np.diff(y, prepend=y[-1])

    angles = np.arctan2(dx, dy)
    angles = np.mod(angles, 2*np.pi)
    da = np.diff(angles, append=angles[0])
    da = np.mod(da+np.pi, 2*np.pi)-np.pi

    return da

from rdp import rdp
import itertools

# def split_contour_by_curvature_old(contour, preview=False):

#     # if contour.ndim == 3: contour = contour[:,0,:] # from opencv, thre is an empty second dimension we can get rid of

#     simplified_contour = rdp(contour, epsilon=0.8)
#     # simplified_contour = filter(contour, 2)

#     da = curvature(simplified_contour)
#     convex_corners = da < -np.pi/6 # if corner is concave with more than 30 degrees

#     convex_corners_indices = convex_corners.nonzero()[0] # get indices of convex corners


#     # loop over all unique combinatinos of two concave corner indixes
#     best_split_contours = None # check all and pick global optimum
#     best_split_separation = MIN_POINT_SEPARATION + 1 # init to some value larger than min separation limit
#     for i0, i1 in itertools.combinations(convex_corners_indices, 2):
        
#         if abs(i0-i1) < 5: continue # valid contours must have more points than this
#         separation = np.linalg.norm(simplified_contour[i0] - simplified_contour[i1])
#         if separation > MIN_POINT_SEPARATION: continue # points muse be closer than this
#         if separation >= best_split_separation: continue # not interested in solution if we have found one with closer points before
        
#         # split to new proposed contours
#         c_a, c_b = split_at_indices(contour=simplified_contour, i0=i0, i1=i1)
        
#         # check areas are sufficiently large
#         if contour_to_area(c_a) < MIN_CELL_AREA: continue
#         if contour_to_area(c_b) < MIN_CELL_AREA: continue

#         best_split_contours = (c_a, c_b)
#         best_split_separation = separation

#     if best_split_contours:
#         c_a, c_b = best_split_contours

#         if preview:
#             plt.figure()
#             plt.title('Curvature')
#             plt.plot(contour[:,0,0], contour[:,0,1], '--')
#             plt.plot(c_a[:,0,0], c_a[:,0,1], '-o')
#             plt.plot(c_b[:,0,0], c_b[:,0,1], '-o')
#             plt.plot(simplified_contour[i0,0,0], simplified_contour[i0,0,1], 'co')
#             plt.plot(simplified_contour[i1,0,0], simplified_contour[i1,0,1], 'co')
#             plt.axis('equal')
        
#         # recurse on sub-contours in case they can be split into more contours
#         return split_contour_by_curvature(c_a, preview=preview) + split_contour_by_curvature(c_b, preview=preview)

#     if preview:
#         plt.figure()
#         plt.plot(contour[:,0,0], contour[:,0,1], '--')
#         plt.axis('equal')

#     return [contour]


# def split_contour_by_curvature2(contour, preview=False):

#     # if contour.ndim == 3: contour = contour[:,0,:] # from opencv, thre is an empty second dimension we can get rid of

#     simplified_contour = rdp(contour, epsilon=0.8)
#     # simplified_contour = filter(contour, 2)

#     da = curvature(simplified_contour)
#     convex_corners = da < -np.pi/6 # if corner is concave with more than 30 degrees

#     convex_corners_indices = convex_corners.nonzero()[0] # get indices of convex corners


#     # loop over all unique combinatinos of two concave corner indixes
#     best_split_contours = None # check all and pick global optimum
#     best_split_separation = MIN_POINT_SEPARATION + 1 # init to some value larger than min separation limit
#     for i0, i1 in itertools.combinations(convex_corners_indices, 2):
        
#         if abs(i0-i1) < 5: continue # valid contours must have more points than this
#         separation = np.linalg.norm(simplified_contour[i0] - simplified_contour[i1])
#         if separation > MIN_POINT_SEPARATION: continue # points muse be closer than this
#         if separation >= best_split_separation: continue # not interested in solution if we have found one with closer points before
        
#         # split to new proposed contours
#         c_a, c_b = split_at_indices(contour=simplified_contour, i0=i0, i1=i1)
        
#         # check areas are sufficiently large
#         if contour_to_area(c_a) < MIN_CELL_AREA: continue
#         if contour_to_area(c_b) < MIN_CELL_AREA: continue

#         best_split_contours = (c_a, c_b)
#         best_split_separation = separation

#     if best_split_contours:
#         c_a, c_b = best_split_contours

#         if preview:
#             plt.figure()
#             plt.title('Curvature')
#             plt.plot(contour[:,0,0], contour[:,0,1], '--')
#             plt.plot(c_a[:,0,0], c_a[:,0,1], '-o')
#             plt.plot(c_b[:,0,0], c_b[:,0,1], '-o')
#             plt.plot(simplified_contour[i0,0,0], simplified_contour[i0,0,1], 'co')
#             plt.plot(simplified_contour[i1,0,0], simplified_contour[i1,0,1], 'co')
#             plt.axis('equal')
        
#         # recurse on sub-contours in case they can be split into more contours
#         return split_contour_by_curvature(c_a, preview=preview) + split_contour_by_curvature(c_b, preview=preview)

#     if preview:
#         plt.figure()
#         plt.plot(contour[:,0,0], contour[:,0,1], '--')
#         plt.axis('equal')

#     return [contour]






# def split_contour(contour, preview=True):
#     da = curvature(contour)
#     pinch_points = np.where(np.abs(da) > np.pi / 6)[0]

#     for i in pinch_points:
#         next_i = (i + 1) % len(contour)
#         c_a = np.concatenate((contour[:i], contour[next_i:]))
#         c_b = contour[i:next_i]
        
#         if contour_to_area(c_a) >= MIN_CELL_AREA and contour_to_area(c_b) >= MIN_CELL_AREA:

#             if preview:
#                 plt.figure()
#                 plt.title('Curvature')
#                 plt.plot(contour[:,0,0], contour[:,0,1], '--')
#                 plt.plot(c_a[:,0,0], c_a[:,0,1], '-o')
#                 plt.plot(c_b[:,0,0], c_b[:,0,1], '-o')
#                 plt.axis('equal')


#             return split_contour(c_a) + split_contour(c_b)
    
#     if preview:
#         plt.figure()
#         plt.title('Curvature')
#         plt.plot(contour[:,0,0], contour[:,0,1], '--')
#         plt.plot(contour[pinch_points,0,0], contour[pinch_points,0,1], 'o')
#         plt.plot(c_a[:,0,0], c_a[:,0,1], '-o')
#         plt.plot(c_b[:,0,0], c_b[:,0,1], '-o')
#         plt.axis('equal')

#     return [contour]


from scipy.interpolate import splprep, splev
from typing import Optional

def compute_curvature(contour: np.ndarray, smoothing: int = 5, preview: bool = False) -> Optional[np.ndarray]:
    # Parametrically represent the contour as a B-spline
    try:
        tck, u = splprep([contour[:,0,0], contour[:,0,1]], s=smoothing, per=True)
    except Exception as e:
        if preview:
            plt.title('Error computing curvature')
            plt.plot(contour[:,0,0], contour[:,0,1], '--')
        return None

    # Derive the B-spline to get the tangent (first derivative)
    dx, dy = splev(u, tck, der=1)
    
    # Derive the B-spline again to get the curvature (second derivative)
    ddx, ddy = splev(u, tck, der=2)
    
    # Compute the curvature
    curvature = (dx * ddy - dy * ddx) / np.power(dx**2 + dy**2, 1.5)
    
    if preview:
        # Evaluate the spline over a range of parameter values for plotting
        new_u = np.linspace(0, 1, len(contour))
        fitted_x, fitted_y = splev(new_u, tck)
        plt.plot(fitted_x, fitted_y, 'r-', label='Fitted B-spline')
    
    return curvature

# def locally_closest_points(contour, pa_index, pb_index, search_distance=5):
#     """
#     Find the closest points on the contour to the points at pa_index and pb_index, respectively,
#     within a limited search range defined by search_distance.
    
#     Parameters:
#         - contour: A 2D numpy array representing the contour.
#         - pa_index, pb_index: Indices of seed points on the contour.
#         - search_distance: The number of points before and after the seed point indices to be considered.
        
#     Returns:
#         - closest_point_to_pa_index, closest_point_to_pb_index: The closest point indices on the contour to pa_index and pb_index, respectively.
#     """
    
#     # Define the search ranges
#     pa_search_range = contour[max(0, pa_index - search_distance) : min(len(contour), pa_index + search_distance + 1)]
#     pb_search_range = contour[max(0, pb_index - search_distance) : min(len(contour), pb_index + search_distance + 1)]
    
#     # Find closest points in the search ranges
    
#     d_matrix = spatial.distance_matrix(x=pa_search_range, y=pb_search_range)
#     # print(d_matrix, rows, cols)

#     ci = np.argmin(d_matrix)
    
#     i1, i2 = np.unravel_index(ci, d_matrix.shape)
#     # print(d_matrix)
#     # print(ci, i1, i2, pa_search_range[i1], pb_search_range[i2])

#     # find index in original contour of i1 and i2
#     closest_point_to_pa_index = np.where(np.all(contour == pa_search_range[i1], axis=1))[0][0]
#     closest_point_to_pb_index = np.where(np.all(contour == pb_search_range[i2], axis=1))[0][0]

#     return closest_point_to_pa_index, closest_point_to_pb_index


def locally_closest_points(contour, pa_index, pb_index, search_distance=5):
    """
    Find the closest points on the contour to the points at pa_index and pb_index, respectively,
    within a limited search range defined by search_distance.
    
    Parameters:
        - contour: A 2D numpy array representing the contour.
        - pa_index, pb_index: Indices of seed points on the contour.
        - search_distance: The number of points before and after the seed point indices to be considered.
        
    Returns:
        - closest_point_to_pa_index, closest_point_to_pb_index: The closest point indices on the contour to pa_index and pb_index, respectively.
    """
    
    # Define the search ranges
    pa_indices_range = np.arange(max(0, pa_index - search_distance), min(len(contour), pa_index + search_distance + 1))
    pb_indices_range = np.arange(max(0, pb_index - search_distance), min(len(contour), pb_index + search_distance + 1))
    
    pa_search_range = contour[pa_indices_range,0,:]
    pb_search_range = contour[pb_indices_range,0,:]
    
    # Find closest points in the search ranges
    d_matrix = spatial.distance_matrix(x=pa_search_range, y=pb_search_range)

    ci = np.argmin(d_matrix)
    
    i1, i2 = np.unravel_index(ci, d_matrix.shape)

    # Get the indices directly from pa_indices_range and pb_indices_range
    closest_point_to_pa_index = pa_indices_range[i1]
    closest_point_to_pb_index = pb_indices_range[i2]

    return closest_point_to_pa_index, closest_point_to_pb_index


def index_separation_in_array(i0, i1, N):
    return min(abs(i0-i1), N-abs(i0-i1))

import scipy.signal as signal
import itertools

MIN_POINT_INDEX_DISTANCE = 10 # 10
MAX_POINT_DISTANCE = 10 # px
MIN_CURVATURE = 0.1 # 0.06


def closest_point_on_other_side_of_contour(point_index: int, contour: np.ndarray, index_separation_limit: int, max_radius: int) -> tuple[int, int]:
    
    close_points_indices = points_within_radius(contour[point_index][0], contour[:,0], max_radius)
    close_points_indices = [j for j in close_points_indices if index_separation_in_array(point_index, j, len(contour)) > index_separation_limit]
    
    if len(close_points_indices) == 0: return None

    distances = np.linalg.norm(contour[point_index][0] - contour[close_points_indices][:,0], axis=1)    
    closest_point_index = close_points_indices[np.argmin(distances)]
    i0, i1 = locally_closest_points(contour, point_index, closest_point_index, search_distance=5)

    if index_separation_in_array(i0, i1, len(contour)) < index_separation_limit : return None # if shifting points made them closer it is not valid
    return i0, i1


# split_factor is a fraction of the maximum width of the contour that is used as the maximum distance between points on opposite sides of the contour when splitting
def split_contour_by_curvature(contour: np.ndarray, split_factor: float, debug: bool=False, printing=False) -> list[np.ndarray]:
    
    if len(contour) < 2*MIN_POINT_INDEX_DISTANCE: return [contour]

    if debug:
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        plt.sca(ax1) # for plot in compute_curvature()

    curvature = compute_curvature(contour, smoothing=10, preview=debug)
    if curvature is None: return [contour]

    # find peaks with amplitude above threshold
    peaks_indices, _ = signal.find_peaks(curvature, height=MIN_CURVATURE, distance=MIN_POINT_INDEX_DISTANCE)
    peak_points = contour[peaks_indices]

    index_separation_limit = np.max([len(contour)//5, MIN_POINT_INDEX_DISTANCE]) # make sure split does not make contour too asymetric

    if debug:
        ax1.plot(contour[:,0,0], contour[:,0,1], 'k-', alpha=0.2)  # 'k-' means black color line
        
        sc = ax1.scatter(contour[:,0,0], contour[:,0,1], c=curvature, cmap='viridis', s=4)
        plt.colorbar(sc, ax=ax1)
        ax1.set_aspect('equal', 'box')

        # Highlight significant positive and negative curvature points
        ax1.plot(peak_points[:,0,0], peak_points[:,0,1], 'ro', markersize=6, label='Pinch points')  # Adjust 0.5 threshold as needed
        
        # Second subplot: Raw Curvature Values
        ax2.set_title(f"Curvature Values")
        ax2.plot(np.maximum(0, curvature), 'g-')
        ax2.hlines(MIN_CURVATURE, 0, len(curvature), colors='r', linestyles='dashed', label='Threshold')
        
        ax2.set_xlabel("Contour Point Index")
        ax2.set_ylabel("Curvature")
        
        plt.tight_layout()


    max_width = max_width_of_countour(contour)
    split_dist = max_width * split_factor

    # find point on other side of contour that is closest -> see if they are close enough
    closest_point_indices = [closest_point_on_other_side_of_contour(i, contour, index_separation_limit, split_dist) for i in peaks_indices]
    closest_point_indices = np.array([i for i in closest_point_indices if i is not None])

    if len(closest_point_indices):
        distances = np.linalg.norm(contour[closest_point_indices[:,0]][:,0] - contour[closest_point_indices[:,1]][:,0], axis=1)
        closest_index = np.argmin(distances)
        i0, i1 = closest_point_indices[closest_index]

        ca, cb = split_at_indices(contour, i0, i1)

        if debug:
            plt.title('Split based on distance')
            ax1.fill(ca[:,0,0], ca[:,0,1], 'r-', alpha=0.2)
            ax1.fill(cb[:,0,0], cb[:,0,1], 'g-', alpha=0.2)
            
            pa = contour[i0][0]
            pb = contour[i1][0]
            ax1.plot([pa[0], pb[0]], [pa[1], pb[1]], 'go', markersize=4, label='Closest')  # Adjust 0.5 threshold as needed
        
        if printing:
            closest_distance = distances[closest_index]
            index_separation = index_separation_in_array(i0, i1, len(contour))
            print('Distance split', len(ca), len(cb), closest_distance, index_separation, i0, i1, len(contour))
        
        return split_contour_by_curvature(contour=ca, split_factor=split_factor, debug=debug) + split_contour_by_curvature(contour=cb, split_factor=split_factor, debug=debug)

    return [contour]

    # Look for pinch points
    
    # compare all combinations of peak points using itertools
    # find the pair with the smallest distance
    closest = None
    closest_distance = np.inf
    for p1i, p2i in itertools.combinations(peaks_indices, 2):
        
        pai, pbi = locally_closest_points(contour, p1i, p2i)
    
        pa, pb = contour[pai][0], contour[pbi][0]
        d = np.linalg.norm(pa - pb)

        index_separation = index_separation_in_array(pai, pbi, len(contour))

        if d < closest_distance and index_separation > index_separation_limit:
            closest_distance = d
            closest = (pai, pbi, pa, pb)

    
    if closest is None: return [contour]
    
    pai, pbi, pa, pb = closest
    
    if closest_distance > MAX_POINT_DISTANCE: return [contour]

    ca, cb = split_at_indices(contour, pai, pbi)
    
    if debug:
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # # First subplot: Contour with Curvature Color Coding
        # ax1.set_title("Contour with Curvature Coloring")
        # ax1.plot(c[:, 0], c[:, 1], 'k-', alpha=0.2)  # 'k-' means black color line
        plt.title('Split based on curvature')
        ax1.fill(ca[:,0,0], ca[:,0,1], 'r-', alpha=0.2)
        ax1.fill(cb[:,0,0], cb[:,0,1], 'g-', alpha=0.2)
        
        ax1.plot([pa[0], pb[0]], [pa[1], pb[1]], 'go', markersize=4, label='Closest')  # Adjust 0.5 threshold as needed
    
    if printing: print('Curvature split', len(ca), len(cb), closest_distance, index_separation, pai, pbi, len(contour))
    return split_contour_by_curvature(contour=ca, split_factor=split_factor, debug=debug) + split_contour_by_curvature(contour=cb, split_factor=split_factor, debug=debug)




def label_contours(contours, frame_size):
    """
    Creates a labeled image where each contour is assigned a unique number.
    
    Args:
    - contours (list): List of contours, where each contour is represented by a list of points.
    - frame_size (tuple): The size of the frame as (height, width).

    Returns:
    - numpy.ndarray: A labeled image with the same size as the frame.
    """
    labeled_img = np.zeros(frame_size, dtype=np.int32)
    
    for idx, contour in enumerate(contours, start=1):
        cv.drawContours(labeled_img, [contour], 0, idx, thickness=cv.FILLED)
        
    return labeled_img


def mask_from_contour(c, padding):
    try:
        c_min = np.min(c[:,0,:],0) - padding
        c_max = np.max(c[:,0,:],0) + padding + 1 # add one to make padding symetrical on all sides
    
    except Exception as e:
        print(e)
        print(c)
        return None, None

    size = c_max-c_min

    mask = np.zeros(size[::-1])
    cv.drawContours(mask, contours=[c-c_min], contourIdx=0, color=1, thickness=cv.FILLED)
    return mask, c_min # c_min is shifting of frame

# # measure max width of cell, in um
# def max_width_of_mask(mask, debug=False):
#     distance_img = ndimage.morphology.distance_transform_edt(mask)
#     return 2*np.max(distance_img) * UM_PER_PIXEL # distance transform gives distance to closest edge -> 2x for full width

# # measure length of skeletons, in um 
# def length_of_mask(mask, debug=False):
#     fil = FilFinder2D(mask, mask=mask)
#     fil.medskel(verbose=False)
#     fil.analyze_skeletons(branch_thresh=400 * u.pix, skel_thresh=2 * u.pix, prune_criteria='length', verbose=False)

#     if debug:
#         plt.figure()
#         plt.imshow(mask)
#         # plt.contour(, colors='r', linewidths=1)
#         plt.contour(fil.skeleton_longpath, colors='r', linewidths=1)
#         plt.axis('off')
#         plt.show()
#         print(fil.branch_lengths())

#     return np.sum(fil.branch_lengths()) * UM_PER_PIXEL

# # Uses skeleton to compute, but is quite slow
# def length_and_width_of_contour(c, debug=False):
#     mask, _ = mask_from_contour(c, padding=1) # smallest padding with complete zero-border
#     return length_of_mask(mask, debug), width_of_mask(mask, debug)

# def lengths_and_widths_of_contours_bounding_box(contours_ts):
#     min_area_rects_ts = twoD_stats(cv.minAreaRect, contours_ts)
#     return [[r[1][0] * UM_PER_PIXEL for r in rs] for rs in min_area_rects_ts], [[r[1][1] * UM_PER_PIXEL for r in rs] for rs in min_area_rects_ts]

# def lengths_and_widths_of_contours(contours_ts):
#     lws = twoD_stats(length_and_width_of_contour, contours_ts)
#     return [[p[0] for p in ps] for ps in lws], [[p[1] for p in ps] for ps in lws]

# def lengths_and_widths_of_contours(contours_ts):
#     a = twoD_stats(length_and_width_of_contour, contours_ts)
#     return list(zip(*[(zip(*b)) for b in a])) # convert list of list of touples to touple of list of lists



def max_width_of_countour(contour) -> float:
    mask, _ = mask_from_contour(contour, padding=1) # smallest padding with complete zero-border
    distance_img = ndimage.morphology.distance_transform_edt(mask)
    return 2*np.max(distance_img)

# Compute all properties at once to avoid re-computing values
def ss_stats_from_contour(contour):
    mask, _ = mask_from_contour(contour, padding=1) # smallest padding with complete zero-border
    distance_img = ndimage.morphology.distance_transform_edt(mask)

    area = cv.contourArea(contour)
    dist_sum = np.sum(distance_img)
    max_width = 2*np.max(distance_img) # distance transform gives distance to closest edge -> 2x for full width
    
    A,E = area, dist_sum
    mean_width_simple = 4*E/A
    mean_width_roots = np.roots([np.pi/48/A, 0, -1/4, E/A])
    mean_width = abs(min(mean_width_roots, key=lambda x: abs(x-mean_width_simple))) # pick root closest to simple solution
    # print(mean_width_roots, mean_width_simple, mean_width)

    length = area/mean_width + mean_width*(1-np.pi/4)
    
    return (
        area * UM_PER_PIXEL**2,
        dist_sum * UM_PER_PIXEL**3,
        length * UM_PER_PIXEL,
        mean_width * UM_PER_PIXEL,
        max_width * UM_PER_PIXEL, 
        length/mean_width,
        length/max_width,
    )

SS_STATS_KEYS = [
    'ss_area',
    'ss_dist_sums',
    'ss_length',
    'ss_width',
    'ss_max_width',
    'ss_aspect_ratio',
    'ss_aspect_ratio_max_width',
    'ss_distance_from_colony_edge',
]

# Helper to convert list of list of touple of stats to dict of list of lists, where dict keys are right for export
def ss_stats_from_contours_ts(contours_ts):
    a = twoD_stats(ss_stats_from_contour, contours_ts)
    b = list(zip(*[(zip(*b)) for b in a])) # convert list of list of touples to touple of list of lists
    return {k: v for k, v in zip(SS_STATS_KEYS, b)}


def id_from_frame_and_outline(f, contour):
    # y,x = centroid(contour)
    # y0,x0 = centroid(contour)
    pl_contour = contour[:,0].astype(float)
    x,y = polylabel_pyo3.polylabel_ext_np(pl_contour, 1.0)
    x,y = round(x), round(y)
    # print(x, x0, '-', y, y0)
    y_max, x_max = f.shape
    if 0 <= y < y_max and 0 <= x < x_max:
        return f[y,x]
    return 0
    
'''
Only keeps top-level contours above threshold size
'''
def contour_filter(cs, min_area):
    
    filter = np.array([contour_to_area(c) > min_area for c in cs])
    
    return [cs[i] for i, included in enumerate(filter) if included]



### Funcitons for extracting stats
    
def oneD_stats(func, timeseries):
    return [func(a) for a in timeseries]

def twoD_stats(func, timeseries):
    #return [np.apply_along_axis(func, 0, a) for a in timeseries]
    return [[func(b) for b in a] for a in timeseries]

### Contour analysis helpers

def contour_to_area(contour):
    return cv.contourArea(contour) * UM_PER_PIXEL**2

def contour_to_arc_length(contour):
    return cv.arcLength(contour, closed=True) * UM_PER_PIXEL

def centroid(contour):
    m = cv.moments(contour)
    m00 = m["m00"] if m["m00"] != 0 else 1 # guard against division by zero
    x = int(m["m01"] / m00)
    y = int(m["m10"] / m00)
    return x, y

def cell_distance_from_colony_border(cell_centroids, colony_contours, frame_shape):
    colony_mask = np.zeros(frame_shape)
    cv.drawContours(colony_mask, contours=colony_contours, contourIdx=-1, color=1, thickness=cv.FILLED)
    colony_edt = ndimage.distance_transform_edt(colony_mask)
    # flatten from multidimentional to 1d array for each index
    return [colony_edt[point[0], point[1]] * UM_PER_PIXEL for point in cell_centroids]

def contours_in_box(contours, bound):
    '''
    return boolean array indicating if relevant conours is fully inside the boinding box.
    boxes defined as [x,y,width,height]
    '''
    
    rects = np.array([cv.boundingRect(c) for c in contours])
    
    left = rects[:,0] > bound[0]
    top = rects[:,1] > bound[1]
    right = rects[:,0] + rects[:,2] < bound[0] + bound[2]
    bottom = rects[:,1] + rects[:,3] < bound[1] + bound[3]
    
    return left * right * top * bottom

def on_border(contour, bound):
    rect = cv.boundingRect(contour)

    left = rect[0] > bound[0] # true if inside left bound
    top = rect[1] > bound[1]
    right = rect[0] + rect[2] < bound[0] + bound[2]
    bottom = rect[1] + rect[3] < bound[1] + bound[3]

    return not (left and top and right and bottom) # if all ar not inside

def contour_intersect(original_image, contour1, contour2):
    # Two separate contours trying to check intersection on
    contours = [contour1, contour2]

    # Create image filled with zeros the same size of original image
    blank = np.zeros(original_image.shape[0:2])

    # Copy each contour into its own image and fill it with '1'
    image1 = cv.drawContours(blank.copy(), contours, 0, 1)
    image2 = cv.drawContours(blank.copy(), contours, 1, 1)

    # Use the logical AND operation on the two images
    # Since the two images had bitwise and applied to them,
    # there should be a '1' or 'True' where there was intersection
    # and a '0' or 'False' where it didn't intersect
    intersection = np.logical_and(image1, image2)

    # Check if there was a '1' in the intersection
    return intersection.any()

# dimensionless number. Circle = 1. Larger number -> less uneven shape/surface.
def perimeter_area_ratio(arc_lengths, areas):
    return arc_lengths**2/areas/(4*np.pi)



### Z-stack related

def focus_score(frame):
    f = cv.GaussianBlur(frame, (5, 5), 0)
    l = cv.Laplacian(f, cv.CV_32S, ksize=5)
    l = np.absolute(l)
    return l.mean()

def focus_score_fft(frame):
    rows, cols = frame.shape
    r_max = min(rows, cols)//2
    frame = frame[:r_max, :r_max]
    frame = np.abs(np.fft.fft2(frame))
    
    # Exploit non-normalized power spectrum for 2d field. 
    # Low frequency components have low area, high frequency have large area
    # Thus, sum becomes weighted in favour of high-frequency components
    return frame.mean() # maximize power spectrum

def best_index_for_stack(stack):
    scores = [focus_score(f) for f in stack]
    return scores.index(max(scores))

def best_indices(stacks):
    return [best_index_for_stack(stack) for stack in stacks]

def best_frames_from_z_stack(stacks):
    if len(stacks[0]) == 1: # no z-stack used, for speedup
        return [s[0] for s in stacks] 

    selection = best_indices(stacks)
    return [stacks[i][idx] for i, idx in enumerate(selection)]

def best_frame_for_stack(movie, stack_indices):
    stack_frames = [movie[int(i)] for i in stack_indices]
    best_index = best_index_for_stack(stack_frames)
    print(f'Checking stack indices {stack_indices}, best is {best_index}')
    return stack_frames[best_index], stack_indices[best_index]

def index_scores(stacks):
    return [[focus_score(f) for f in stack] for stack in stacks]

# def z_stack_series(m, start, number_of_stacks=None, z_stack_size=7, consecutive_images=2, interval=2*7*60*4):
#     indices = range(start, len(m), interval)
#     if number_of_stacks:
#         indices = indices[:number_of_stacks]

#     stacks = [m[i:i+consecutive_images*z_stack_size:consecutive_images] for i in indices] # z-stacks
#     times = [m.frame_time(i, true_time=True) for i in indices]
#     return stacks, times, indices



WINDOW_SIZE = 200 # 200 is close to GCD of 3208x2200, 240 is gcd of pixel counts (1920x1200)
BLUR_SIZE = 51

class MemoizeKernelWindow:
    def __init__(self, f):
        self.f = f
        self.memo = {}
    def __call__(self, xs, ys, shape):
        key = (xs.start, xs.stop, ys.start, ys.stop, shape)
        if not key in self.memo:
            self.memo[key] = self.f(xs, ys, shape)
        return self.memo[key]

@MemoizeKernelWindow
def window_kernel(xs, ys, shape):
    k = np.zeros(shape).astype(np.uint16)
    k[ys,xs] = 255
    # k = cv.GaussianBlur(k, (51, 51), 0)
    k = cv.blur(k, (BLUR_SIZE, BLUR_SIZE))
    k = k.astype(np.float16)/255.0
    # plot_frame(k.astype(np.uint16), 'kernel', plot=xs.start == 0 and ys.start == 0)
    return k


### Colors


def color_for_number(number):
    return ColorHash(number).rgb if number != -1 else (0,0,255)
    



### Plotting


def plot_frame(f, dinfo, contours=None, new_figure=True, contour_thickness=1, contour_color_function=None, contour_labels=None):
    
    if not dinfo.live_plot and not dinfo.file_plot: # for performance
        return 
    
    f = to_dtype_uint8(f)

    if contours:
        f = np.stack((f,)*3, axis=-1)
        for i, _ in enumerate(contours):
            color = color_for_number(i) if contour_color_function == None else contour_color_function(i, contours[i])
            f = cv.drawContours(f, contours, contourIdx=i, color=color, thickness=contour_thickness)

        if contour_labels:
            for label, contour in zip(contour_labels, contours):
                y,x = centroid(contour)
                f = text_on_frame(f, label, (x+10,y+10))

    if dinfo.crop != None:
        (x0,x1), (y0,y1) = dinfo.crop
        f = f[y0:y1, x0:x1]

    if dinfo.live_plot:
        if new_figure:
            plt.figure(figsize=(25, 10), dpi=80)

        plt.imshow(f, cmap='gray')
        plt.title(dinfo.label, color='w')
        plt.xticks([]), plt.yticks([])
        plt.tight_layout()

    if dinfo.file_plot:
        im = Image.fromarray(f)
        im.save(os.path.join(dinfo.image_dir, dinfo.label + '.png'))
    

'''
Plot masks with color corresponding to cell area
'''
def plot_frame_color_area(f, dinfo, contours):
    
    max_area = 1 if len(contours) == 0 else np.max([contour_to_area(contour) for contour in contours])
    def area_color(index, contour): 
        area = contour_to_area(contour)
        return (area*255/max_area, 0, 255-area*255/max_area) # r,g,b
    
    plot_frame(f, dinfo=dinfo, contours=contours, contour_thickness=cv.FILLED, contour_color_function=area_color)

'''
Plot masks with color corresponding to distance from colony edge
'''
def plot_frame_color_edist(f, dinfo, cell_contours, colony_contours):
    
    cell_centroids = oneD_stats(centroid, cell_contours)
    cell_distance = cell_distance_from_colony_border(cell_centroids, colony_contours, f.shape)
    max_dist = np.max(cell_distance) if len(cell_distance) else 1
    def edge_distance_function(index, contour):
        d = cell_distance[index]
        return (d*255/max_dist, 0, 255-d*255/max_dist) # r,g,b
    
    plot_frame(f, dinfo=dinfo, contours=cell_contours, contour_thickness=cv.FILLED, contour_color_function=edge_distance_function)




def frame_with_cs_ss_offset(frame, cs_contours, cs_ids, ss_contours, ss_ids, offset, cs_on_border, ss_stroke=1):
    f = norm(frame).astype(np.uint8)
    f = np.stack((f,)*3, axis=-1)

    for i, c, b in zip(cs_ids, cs_contours, cs_on_border):
        cv.drawContours(f, contours=[c], contourIdx=0, color=color_for_number(i), thickness=8 if b else 2)
    for i, c in zip(ss_ids, ss_contours):
        cv.drawContours(f, contours=[c], contourIdx=0, color=color_for_number(i), thickness=ss_stroke)
    
    if not isinstance(offset, type(None)):
        offset = np.append(offset, [0]) # add third rgb dimension
        f = scipy.ndimage.shift(f, shift=offset, cval=0)

    return f

'''
Plots entire stack at plt from matplotlib.
TODO: make file output from matplotlib figures, not just single frames as now
'''
def plot_stack(fs, dinfo):
    fig = plt.figure(figsize=(50, 10), dpi=180)
    gs = fig.add_gridspec(1, len(fs), hspace=0, wspace=0)

    scores = [focus_score(f) for f in fs]
    scores2 = [focus_score_fft(f) for f in fs]
    labels = [f'{i}: {s:.1e}, {s2:.1e}   {"b1" if max(scores)==s else ""} {"b2" if max(scores2)==s2 else ""}' for i, (s, s2) in enumerate(zip(scores, scores2))]

    for label, f, ax in zip(labels, fs, gs.subplots(sharey='row')):
        plt.sca(ax)
        plot_frame(f, dinfo.append_to_label(label), new_figure=False)



'''
Plot contour without background image
'''
def plot_contour(c):
    xs = c[:,0]
    ys = c[:,1]
    xs = np.append(xs, xs[:1])
    ys = np.append(ys, ys[:1])

    plt.gca().set_aspect('equal', 'box')
    plt.plot(xs,ys, 'o-')


### Debug visualizations

@cache
def get_font():
    try:
        font = ImageFont.truetype("arial.ttf", 30)
        logging.info('Using font arial.ttf')
    except OSError:
        try:
            font = ImageFont.truetype("Lato-Regular.ttf", 40)
            logging.info('Using font Lato-Regular.ttf')
        except OSError:
            font = ImageFont.load_default()
            logging.info('Using font default font')
    
    return font

'''
Add text to frame
'''
def text_on_frame(frame, text, position):
    image = Image.fromarray(frame, 'RGB' if len(frame.shape) == 3 else None)
    canvas = ImageDraw.Draw(image)
    font = get_font()
    
    canvas.text(position, f'{text}', (225, 225, 225), font=font)
    return np.asarray(image)


def translate_contours(contours, offset):
    return [contour+offset for contour in contours]


def masks_to_movie(frames_ts, cs_contours_ts, cs_ids_ts, ss_contours_ts, ss_ids_ts, cumulative_offset_ts, frame_labels, times, ss_stroke, dinfo, output_frames=False):
        FPS = 5
        out = cv.VideoWriter(os.path.join(dinfo.video_dir, f'{dinfo.label}.mp4'), cv.VideoWriter_fourcc(*'mp4v'), FPS, frames_ts[0].shape[::-1])

        logging.debug(f'{len(frames_ts)}, {len(cs_contours_ts)}, {len(cs_ids_ts)}, {len(ss_contours_ts)}, {len(ss_ids_ts)}, {len(cumulative_offset_ts)}, {len(times)}')

        ### Generate frames with colony and single-cell annotation
        for f, cs_contours, cs_ids, ss_contours, ss_ids, cumulative_offset, frame_label, time \
            in zip(frames_ts, cs_contours_ts, cs_ids_ts, ss_contours_ts, ss_ids_ts, cumulative_offset_ts, frame_labels, times):
            debug_frame = frame_with_cs_ss_offset(
                frame=f, 
                cs_contours=cs_contours, 
                cs_ids=cs_ids,
                ss_contours=ss_contours, 
                ss_ids=ss_ids,
                offset=cumulative_offset,
                cs_on_border=[False]*len(cs_contours),
                ss_stroke=ss_stroke,
            )

            ### Add text to frames
            debug_label = f'{dinfo.label}, {frame_label}'
            time_label = f'{time//(60*60):02.0f}:{(time//60) % 60:02.0f}:{time % 60:02.0f}'
            time_label_full = f'time = {time_label}'
            debug_frame = text_on_frame(debug_frame, debug_label, position=(10, 20))
            debug_frame = text_on_frame(debug_frame, time_label_full, position=(10, 50))

            out.write(debug_frame)

            if output_frames: 
                # output full frame
                time_label = f'full_t{time_label.replace(":", ".")}'
                plot_frame(debug_frame, dinfo.append_to_label(time_label))

                debug_frame_no_offset = frame_with_cs_ss_offset(
                    frame=f, 
                    cs_contours=cs_contours, 
                    cs_ids=cs_ids,
                    ss_contours=ss_contours, 
                    ss_ids=ss_ids,
                    offset=[0 for f in cumulative_offset],
                    cs_on_border=[False]*len(cs_contours),
                    ss_stroke=ss_stroke,
                )

                # output slice of frame that follows colony
                for contour, id in zip(cs_contours, cs_ids):
                    if on_border: continue # skip colonies on border
                    
                    PADDING = 10 # px
                    x,y,w,h = cv.boundingRect(contour)
                    plot_frame(debug_frame_no_offset[y-PADDING:y+h+PADDING, x-PADDING:x+w+PADDING], dinfo.append_to_label(f'cid{id}_{frame_label}_{time_label}'))

        out.release()



COLONY_OUTPUT_SIZE = 128

def export_masks_for_first_generation(cs_contours_ts, names_ts, frames_ts, label, output_folder):
    
    if output_folder == None: return
    
    completed = set()
    for cs_contours, names, frame in zip(cs_contours_ts, names_ts, frames_ts):
        for cs, name in zip(cs_contours, names):
            if '.' in name: continue
            if name in completed: continue

            completed.add(name)
            
            # make image mask with ones and zeros
            mask = np.zeros_like(frame)
            cv.drawContours(mask, contours=[cs], contourIdx=0, color=1, thickness=cv.FILLED)
            
            # and with frame
            masked_full_frame = np.multiply(frame, mask)
            x,y,w,h = cv.boundingRect(cs)
            
            output_frame = np.zeros((COLONY_OUTPUT_SIZE, COLONY_OUTPUT_SIZE))
            
            x0, y0 = (COLONY_OUTPUT_SIZE-w)//2, (COLONY_OUTPUT_SIZE-h)//2
            if x0 < 0:
                logging.debug(f'Cannot fit mask in x for name {name} in frame, outline is {x0=}, {x=}, {w=}')
                x += (w-COLONY_OUTPUT_SIZE)//2
                x0, w = 0, COLONY_OUTPUT_SIZE
                logging.debug(f'After update, {name} outline is {x0=}, {x=}, {w=}')
            if y0 < 0:
                logging.debug(f'Cannot fit mask in x for name {name} in frame, outline is {y0=}, {y=}, {h=}')
                y += (h-COLONY_OUTPUT_SIZE)//2
                y0, h = 0, COLONY_OUTPUT_SIZE
                logging.debug(f'After update, {name} outline is {y0=}, {y=}, {h=}')
            try:
                output_frame[y0:y0+h, x0:x0+w] = masked_full_frame[y:y+h, x:x+w]

                im = Image.fromarray(np.uint8(output_frame))
                im.save(os.path.join(output_folder, f'{label}_n{name}_x{x}_y{y}.jpg'))
            
            except:
                logging.exception('Could not generate mask frame')