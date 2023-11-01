
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
import numpy as np
import cv2 as cv

from PadAnalyser.MicrocolonySegmenter import MKSegmentUtils


kc5 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5)).astype(np.uint8)


# Dilate contour by converting to mask, dilating and converting back to contour
def dilate_contour(c):
    mask, c_min = MKSegmentUtils.mask_from_contour(c, padding=5)
    mask = cv.dilate(mask, kernel=MKSegmentUtils.k5_circle, iterations=1)
    ca, _ = cv.findContours(mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return ca[0]+c_min
    

def filter_to_colonies(m1, colony_contours):
    colony_mask = np.zeros_like(m1, dtype=np.uint8)
    cv.drawContours(colony_mask, colony_contours, -1, 1, cv.FILLED)
    return m1 * colony_mask

def laplacian_uint8(f):

    assert f.dtype == np.uint8

    laplacian_frame = cv.GaussianBlur(f, (7, 7), 0) # blur, kernel size about feature size
    laplacian_frame = cv.Laplacian(laplacian_frame, cv.CV_16S, ksize=7) # laplacian
    laplacian_frame = laplacian_frame.astype(np.int16)
    
    return laplacian_frame


def laplacian_of_gaussian(image, sigma, ksize):
    blurred = cv.GaussianBlur(image, (0, 0), sigma)
    return cv.Laplacian(blurred, ddepth=cv.CV_64F, ksize=ksize)


def label_contour_in_mask(mask, dinfo):
    
    # Apply morphological operations with a larger structuring element
    # selem = disk(1)  # Adjust the size as necessary
    # eroded = erosion(mask, selem)
    # MKSegmentUtils.plot_frame(eroded, dinfo=dinfo.append_to_label('1_eroded'))

    # labeled_regions, num_features = ndimage.label(mask)

    # Apply distance transform
    distance_transform = ndimage.distance_transform_edt(mask)
    MKSegmentUtils.plot_frame(distance_transform>2, dinfo=dinfo.append_to_label('3.0_dt_g'))
    MKSegmentUtils.plot_frame(distance_transform, dinfo=dinfo.append_to_label('3.1_dt_raw'))
    
    # Apply smoothing on the distance map
    distance_transform = gaussian(distance_transform, sigma=0.5)  # Adjust the sigma value as necessary
    MKSegmentUtils.plot_frame(distance_transform, dinfo=dinfo.append_to_label('3.2_dt_filtered'))

    h_maxima = [extrema.h_maxima(distance_transform, i).astype(bool) for i in [1,2,3,4]]

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



def robust_edge_detection(gray_image, lower_threshold=50, upper_threshold=150):
    
    # Apply Gaussian blur
    blurred_image = gray_image # cv2.GaussianBlur(gray_image, (3, 3), 0)
    
    # Apply the Canny edge detector
    edges = cv.Canny(blurred_image, lower_threshold, upper_threshold)

    # # Dilate the edges to make them thicker
    # dilated_edges = cv2.dilate(edges, None, iterations=3)  # Increase the iterations if necessary
    
    # # Erode to bring them back to a similar width but ensure thickness
    # final_edges = cv2.erode(dilated_edges, None, iterations=3)  # Match the iterations used in dilation
    final_edges = edges

    return final_edges

