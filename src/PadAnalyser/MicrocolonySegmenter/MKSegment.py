from . import MKSegmentUtils, DInfo, ZStack
import numpy as np
import cv2 as cv
import logging
import skimage.transform
import scipy.signal


# simple flatten stack with stitching 
def flatten_stack(stack, dinfo):

    plane_coefficients = None
    
    # Check if stack is already flattened, otherwise compute projection
    if isinstance(stack, np.ndarray): frame_raw = stack
    elif len(stack) == 1: frame_raw = stack[0]
    else: frame_raw, plane_coefficients = ZStack.project_to_plane(stack, dinfo=dinfo) # compute laplacian from normalized frame

    frame = MKSegmentUtils.norm(frame_raw)
    # frame = MKSegmentUtils.to_dtype_uint8(frame_raw)

    # compute laplacian compressed stack
    laplacian_frame = cv.GaussianBlur(frame_raw, (7, 7), 0) # blur, kernel size about feature size
    laplacian_frame = cv.Laplacian(laplacian_frame, cv.CV_32S, ksize=7) # laplacian
    laplacian_frame = laplacian_frame//2**16 # scale to fit in int16
    laplacian_frame = laplacian_frame.astype(np.int16)

    # output debug frames
    MKSegmentUtils.plot_frame(frame_raw, dinfo=dinfo.append_to_label('z_stack_best_raw'))
    MKSegmentUtils.plot_frame(frame, dinfo=dinfo.append_to_label('z_stack_best'))
    MKSegmentUtils.plot_frame(laplacian_frame, dinfo=dinfo.append_to_label(f'z_stack_laplacian'))
    # for i, s in enumerate(stack):
    #     MKSegmentUtils.plot_frame(s, dinfo=dinfo.append_to_label(f'z_stack_frame_{i}'))

    return frame, laplacian_frame, plane_coefficients


from skimage import measure #, morphology, filters
# from skimage.morphology import disk, erosion
from skimage.filters import gaussian
from skimage.segmentation import watershed
from scipy import ndimage
# from skimage.feature import peak_local_max
# from skimage.segmentation import random_walker
# from scipy.ndimage import gaussian_laplace
from skimage.morphology import extrema
from scipy.ndimage import convolve

def label_contour_in_mask(mask, dinfo):
    
    # Apply morphological operations with a larger structuring element
    # selem = disk(1)  # Adjust the size as necessary
    # eroded = erosion(mask, selem)
    # MKSegmentUtils.plot_frame(eroded, dinfo=dinfo.append_to_label('1_eroded'))

    # labeled_regions, num_features = ndimage.label(mask)

    # Apply distance transform
    distance_transform = ndimage.distance_transform_edt(mask)
    MKSegmentUtils.plot_frame(distance_transform, dinfo=dinfo.append_to_label('3.1_dt_raw'))
    
    # Apply smoothing on the distance map
    distance_transform = gaussian(distance_transform, sigma=0.5)  # Adjust the sigma value as necessary
    MKSegmentUtils.plot_frame(distance_transform, dinfo=dinfo.append_to_label('3.2_dt_filtered'))

    h_maxima = [extrema.h_maxima(distance_transform, i).astype(bool) for i in [1]]

    # Label each separate region in the binary image
    labeled_regions, num_features = ndimage.label(mask)

    # Initialize an empty array to store the final maxima
    final_maxima = np.zeros_like(mask, dtype=bool)
    
    mean_area = np.sum(mask) / num_features

    # Issue: 
    # - when cells are large, I preffer high h_maxima to keep fragments together
    # - when cells are small, I preffer low h_maxima to split fragments that have been segmented as one
    # Solution: 
    # print(f'Mean cell area: {mean_area}')
    prefer_small_cells = mean_area < 300

    radii = []
    areas = []

    for i in range(1, num_features + 1):
        region = (labeled_regions == i)
        distance_transform_region = distance_transform[region]
        
        area = np.sum(region)
        radius = np.max(distance_transform_region)
        
        if radius < 1: # or area < 12:
            continue
        
        radii.append(radius)
        areas.append(area)

        for h_maxima_i in h_maxima if prefer_small_cells else reversed(h_maxima):
            if np.any(h_maxima_i & region):
                final_maxima |= (h_maxima_i & region)
                break
        
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(6,4), dpi=200)
    # plt.hist(radii, bins=40)
    # plt.title(f'Radii {dinfo.label}')
    # plt.figure(figsize=(6,4), dpi=200)
    # plt.hist(areas, bins=40)
    # plt.title(f'Areas {dinfo.label}')

    # convolution to make diagonal pixels make up a continous region
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    final_maxima = convolve(final_maxima>0, kernel) > 0

    markers = ndimage.label(final_maxima)[0]
    MKSegmentUtils.plot_frame((mask>0).astype(np.uint8)*100 + (markers>0).astype(np.uint8)*155, dinfo=dinfo.append_to_label('3.3_markers'))

    # Apply watershed transformation using the distance map
    labels_ws = watershed(-distance_transform, markers, mask=mask)
    
    MKSegmentUtils.plot_frame(labels_ws, dinfo=dinfo.append_to_label('3.4_labels_ws'))

    labeled_cells = measure.label(labels_ws)
    MKSegmentUtils.plot_frame(labeled_cells, dinfo=dinfo.append_to_label('3.5_labelled'))

    ### Compute contours - move away from this? 
    # Get unique labels
    unique_labels = np.unique(labeled_cells)

    # List to store contours
    all_contours = []
    contour_labels = []

    # Find contours for each labeled region
    for label in unique_labels:
        if label == 0:
            continue  # Skip background label
        mask = np.zeros_like(labeled_cells, dtype=np.uint8)
        mask[labeled_cells == label] = 255
        
        # Find contours
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        all_contours.extend(contours)

        # radius = np.max(distance_transform[mask])
        # contour_labels.append(f'{radius:.1f}')

    # MKSegmentUtils.plot_frame(labeled_cells, dinfo=dinfo.append_to_label('07_contour_radius'), contours=contours, contour_thickness=cv.FILLED, contour_labels=contour_labels)

    return all_contours


'''Assume frame has been preprocessed, and is uint16'''
def bf_single_cell_segment(f, colony_contours, dinfo):

    # make frame from colony contours
    colony_mask = np.zeros_like(f, dtype=np.uint8)
    cv.drawContours(colony_mask, colony_contours, -1, 1, cv.FILLED)

    if dinfo.printing:
        logging.debug(f'Raw frame info: type={f.dtype}, min={np.min(f)}, max={np.max(f)}')

    ### Include intensity from BF to get rid of cell outlines -> m1
    m1 = f
    m1 = MKSegmentUtils.norm(f)
    m1 = cv.GaussianBlur(m1, (3, 3), 0)
    # for o in [-3,-4,-5,-6,-8]:
    # for a in [11, 15, 21, 41, 61, 81]:
    #     m10 = cv.adaptiveThreshold(m1, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, a, -4)
    #     MKSegmentUtils.plot_frame(m10, dinfo=dinfo.append_to_label(f'3_m1_{a}'))

    m1 = cv.adaptiveThreshold(m1, 1, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 81, -4) # -4, relativley low, to make better at segmenting big cells such as Mecilinam-exposed ecoli
    MKSegmentUtils.plot_frame(m1*255, dinfo=dinfo.append_to_label('3_m1'))

    # remove background based on colony locations
    m1 = m1 * colony_mask

    # remove background lonely pixels 
    # kernel = np.ones((5, 5))
    # isolated_pixels = convolve(m1, kernel) < 5
    # m1 = m1 & ~isolated_pixels

    MKSegmentUtils.plot_frame(m1*255, dinfo=dinfo.append_to_label('3_m2'))

    ## Detect contours -> contours
    contours = label_contour_in_mask(m1, dinfo=dinfo.append_to_label('4_lbl'))
    MKSegmentUtils.plot_frame(f, dinfo=dinfo.append_to_label('07_contours_all'), contours=contours, contour_thickness=cv.FILLED)

    contours = [MKSegmentUtils.dilate_contour(c) for c in contours]
    MKSegmentUtils.plot_frame(f, dinfo=dinfo.append_to_label('11_contours_dilated'), contours=contours, contour_thickness=cv.FILLED)

    return contours


    # m1 = cv.adaptiveThreshold(m1, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 81, -4)
    # MKSegmentUtils.plot_frame(m1, dinfo=dinfo.append_to_label('3_m11'))
    
    # mf = m1
    # contours, hierarchy = cv.findContours(np.uint8(mf), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # MKSegmentUtils.plot_frame(f, dinfo=dinfo.append_to_label('07_contours_all'), contours=contours)

    # # contours = MKSegmentUtils.contour_filter(contours, hierarchy, min_area=MKSegmentUtils.MIN_CELL_AREA)
    # # MKSegmentUtils.plot_frame(f, dinfo=dinfo.append_to_label('08_contours_filtered'), contours=contours, contour_thickness=cv.FILLED)    

    # # contours = [c_out for c_in in contours for c_out in MKSegmentUtils.split_contour_by_point_distance(c_in, preview=False)]
    # # MKSegmentUtils.plot_frame(f, dinfo=dinfo.append_to_label('09_contours_split_point_distance'), contours=contours, contour_thickness=cv.FILLED)
    
    # # contours = [c_out for c_in in contours for c_out in MKSegmentUtils.split_contour_by_curvature(c_in, preview=False)]
    # # MKSegmentUtils.plot_frame(f, dinfo=dinfo.append_to_label('10_contours_split_curvature'), contours=contours, contour_thickness=cv.FILLED)
    
    # # contours = [MKSegmentUtils.dilate_contour(c) for c in contours]
    # # MKSegmentUtils.plot_frame(f, dinfo=dinfo.append_to_label('11_contours_dilated'), contours=contours, contour_thickness=cv.FILLED)

    # ### Done! -> contours
    # return contours


def bf_colony_segment(l, dinfo):
        
    ### Colony outline from laplacian -> m0
    m0 = l
    # for t in [1,2,3,4,5,10]:
    #     for s in [-1,-2,-3, -4, -5, -10]:
    #         M00 =  np.logical_or(s > M0, M0 > t)
    #         MKSegmentUtils.plot_frame(M00, title=f'{label}_col_2_M00_{s}_{t}', plot=plot, dir=debug_dir, crop=crop)

    m0 =  np.logical_or(-5 >= m0, m0 >= 5)
    # m0 =  m0 < -8
    MKSegmentUtils.plot_frame(m0, dinfo=dinfo.append_to_label('2_m0'))

    ### Close borders -> m2
    m2 = m0
    m2 = scipy.signal.convolve2d(m2, MKSegmentUtils.k9_circle, mode='same') > np.sum(MKSegmentUtils.k9_circle)*1/2
    m2 = np.logical_or(m2, m0)
    m2 = scipy.signal.convolve2d(m2, MKSegmentUtils.k9_circle, mode='same') > np.sum(MKSegmentUtils.k9_circle)*1/2
    m2 = np.logical_or(m2, m0)
    MKSegmentUtils.plot_frame(m2, dinfo=dinfo.append_to_label('3_m2'))

    # ### Filter any elements smaller than n blocks -> m1
    m1 = m2
    # m1 = scipy.signal.convolve2d(m1, MKSegmentUtils.kL1, mode='same') > np.sum(MKSegmentUtils.kL1)//2
    # m1 = np.logical_and(m1, m2)
    # MKSegmentUtils.plot_frame(m1, dinfo=dinfo.append_to_label('4_m1'))
    
    ### Fill holes -> mu
    mu = m1
    # mu = cv.morphologyEx(mu.astype(np.uint8), cv.MORPH_CLOSE, MKSegmentUtils.kL2.astype(np.uint8)) # fill last remaining holes
    # MKSegmentUtils.plot_frame(mu.astype(bool), dinfo=dinfo.append_to_label('5_mu'))
    # mu = cv.morphologyEx(mu.astype(np.uint8), cv.MORPH_OPEN, MKSegmentUtils.kL2.astype(np.uint8)) # harsh debris removal
    # MKSegmentUtils.plot_frame(mu.astype(bool), dinfo=dinfo.append_to_label('6_mu'))
    mu = cv.GaussianBlur(mu.astype(np.uint8), (41, 41), 0)
    # mu = cv.morphologyEx(mu.astype(np.uint8), cv.MORPH_DILATE, k5_circle.astype(np.uint8)) # make mask a bit bigger
    MKSegmentUtils.plot_frame(mu.astype(bool), dinfo=dinfo.append_to_label('7_mu'))


    ### Contour detection -> ca
    # contours = label_contour_in_mask(mu, dinfo=dinfo.append_to_label('8_lbl'))
    # MKSegmentUtils.plot_frame(l, dinfo=dinfo.append_to_label('8_contours_all'), contours=contours, contour_thickness=2)

    # return contours

    contours, hierarchy = cv.findContours(np.uint8(mu), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    MKSegmentUtils.plot_frame(l, dinfo=dinfo.append_to_label('8_contours_all'), contours=contours, contour_thickness=2)
    
    contours = MKSegmentUtils.contour_filter(contours, hierarchy, min_area=MKSegmentUtils.MIN_COLONY_AREA)
    MKSegmentUtils.plot_frame(l, dinfo=dinfo.append_to_label('9_contours_filtered'), contours=contours, contour_thickness=2)

    ## Done! -> contours
    return contours
