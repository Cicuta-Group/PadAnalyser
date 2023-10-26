
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
import cv2 as cv


# Dilate contour by converting to mask, dilating and converting back to contour
def dilate_contour(c):
    mask, c_min = mask_from_contour(c, padding=5)
    mask = cv.dilate(mask, kernel=k5_circle, iterations=1)
    ca, _ = cv.findContours(mask.astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return ca[0]+c_min
    






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
