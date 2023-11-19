
from . import MKSegmentUtils, DInfo, CellSegmentMods
import numpy as np
import cv2 as cv
import scipy.signal
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import os

from skimage import segmentation

MIN_EDGE_DISTANCE = 20



def laplaican_profile(laplacian, c):
    
    # Create a mask for the current contour
    mask = np.zeros_like(laplacian, dtype=np.uint8)
    cv.drawContours(mask, [c], -1, 1, -1)  # -1 fills the contour

    # Calculate the distance transform
    dist_transform = cv.distanceTransform(mask, cv.DIST_L2, 3)

    # Round distances to nearest integer to get integer resolution
    int_dist_transform = np.round(dist_transform).astype(np.int32)

    # Create an array to hold the sum of Laplacian values for each distance
    max_dist = min(12, np.max(int_dist_transform))

    def get_laplacian_mean(dist):
        mask_distance = int_dist_transform == dist
        return np.mean(laplacian * mask_distance)
    
    distances = range(1, max_dist + 1)
    means = [get_laplacian_mean(d) for d in distances]
    return means


def filter_contours_by_laplacian_profile(laplacian, contours, dinfo, annotated_mask=None):

    colony_profile = [laplaican_profile(laplacian, c) for c in contours] # list of lists of laplacian values as function of distance from edge of colony
    contour_is_colony = [check_if_colony_profile(p) for p in colony_profile]
    
    if dinfo.live_plot or dinfo.file_plot:
        plt.figure(figsize=(6,4), dpi=300)
        if annotated_mask is None:
            for i, (profile, is_colony) in enumerate(zip(colony_profile, contour_is_colony)):
                plt.plot(profile, '-' if is_colony else '--', label=f'Contour {i}')
        else:
            colonies_are_annotated = [check_if_cell_in_colony(annotated_mask, c) for c in contours]
            for i, (profile, is_colony, colony_is_annotated) in enumerate(zip(colony_profile, contour_is_colony, colonies_are_annotated)):
                match = is_colony == colony_is_annotated
                plt.plot(profile, '-' if colony_is_annotated else '--', color='black' if match else 'red', label=f'Contour {i}')
        plt.xlabel('Distance from edge')
        plt.ylabel('Average Laplacian value')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(dinfo.label)
        if dinfo.file_plot: plt.savefig(os.path.join(dinfo.image_dir, f'{dinfo.label}.png'), bbox_inches='tight')
        if not dinfo.live_plot: plt.close()

    return [c for c, is_colony in zip(contours, contour_is_colony) if is_colony]


def check_if_cell_in_colony(annotated_cell_mask, c_contour):
    mask = np.zeros_like(annotated_cell_mask, dtype=np.uint8)
    cv.drawContours(mask, [c_contour], -1, 1, -1)  # -1 fills the contour
    return np.any(annotated_cell_mask * mask)


def check_if_colony_profile(vector, value_threshold=0.05, difference_threshold=0.1):
    """
    Checks the conditions on the vector for max and min values and their indices.

    :param vector: The input vector with length <= 10
    :param value_threshold: The minimum value the max must be above
    :param difference_threshold: The minimum difference required between max and min values
    :return: A tuple (bool, max_value, max_index, min_value, min_index)
             bool - True if all conditions are met, False otherwise
             max_value - The maximum value in the vector
             max_index - The index of the maximum value
             min_value - The minimum value in the vector
             min_index - The index of the minimum value
    """

    # Find the max and min values and their indices
    max_value = max(vector)
    min_value = min(vector)
    max_index = vector.index(max_value)
    min_index = vector.index(min_value)

    # Check the conditions
    return \
        max_index < min_index and \
        max_value > value_threshold and \
        (max_value - min_value) > difference_threshold




# def bf_colony_segment(f, dinfo: DInfo):
    
#     l = CellSegmentMods.laplacian_uint8(f)

#     ### Colony outline from laplacian -> m0
#     m0 = l
#     # for t in [1,2,3,4,5,10]:
#     #     for s in [-1,-2,-3, -4, -5, -10]:
#     #         M00 =  np.logical_or(s > M0, M0 > t)
#     #         MKSegmentUtils.plot_frame(M00, title=f'{label}_col_2_M00_{s}_{t}', plot=plot, dir=debug_dir, crop=crop)

#     THRESHOLD = 2000
#     m0 =  np.logical_or(-THRESHOLD >= m0, m0 >= THRESHOLD)
#     # m0 =  m0 < -8
#     MKSegmentUtils.plot_frame(m0, dinfo=dinfo.append_to_label('2_m0'))

#     ### Close borders -> m2
#     m2 = m0
#     m2 = scipy.signal.convolve2d(m2, MKSegmentUtils.k9_circle, mode='same') > np.sum(MKSegmentUtils.k9_circle)*1/2
#     m2 = np.logical_or(m2, m0)
#     MKSegmentUtils.plot_frame(m2, dinfo=dinfo.append_to_label('3_m2'))

#     # ### Filter any elements smaller than n blocks -> m1
#     m1 = m2
#     m1 = scipy.signal.convolve2d(m1, MKSegmentUtils.kL1, mode='same') > np.sum(MKSegmentUtils.kL1)//2
#     m1 = np.logical_and(m1, m2)
#     MKSegmentUtils.plot_frame(m1, dinfo=dinfo.append_to_label('4_m1'))
    
#     ### Fill holes -> mu
#     mu = m1
#     mu = cv.morphologyEx(mu.astype(np.uint8), cv.MORPH_CLOSE, MKSegmentUtils.kL2.astype(np.uint8)) # fill last remaining holes
#     MKSegmentUtils.plot_frame(mu.astype(bool), dinfo=dinfo.append_to_label('5_mu'))
#     mu = cv.morphologyEx(mu.astype(np.uint8), cv.MORPH_OPEN, MKSegmentUtils.kL2.astype(np.uint8)) # harsh debris removal
#     MKSegmentUtils.plot_frame(mu.astype(bool), dinfo=dinfo.append_to_label('6_mu'))
#     mu = cv.GaussianBlur(mu, (21, 21), 0)
#     # mu = cv.morphologyEx(mu.astype(np.uint8), cv.MORPH_DILATE, k5_circle.astype(np.uint8)) # make mask a bit bigger
#     MKSegmentUtils.plot_frame(mu.astype(bool), dinfo=dinfo.append_to_label('7_mu'))

#     mu = segmentation.clear_border(mu)

#     ### Contour detection -> ca
#     # contours = label_contour_in_mask(mu, dinfo=dinfo.append_to_label('8_lbl'))
#     # MKSegmentUtils.plot_frame(l, dinfo=dinfo.append_to_label('8_contours_all'), contours=contours, contour_thickness=2)

#     # return contours

#     # filter regions touching border
    


#     contours, _  = cv.findContours(np.uint8(mu), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#     MKSegmentUtils.plot_frame(l, dinfo=dinfo.append_to_label('8_contours_all'), contours=contours, contour_thickness=2)
    
#     contours = MKSegmentUtils.contour_filter(contours, min_area=MKSegmentUtils.MIN_COLONY_AREA)
#     MKSegmentUtils.plot_frame(l, dinfo=dinfo.append_to_label('9_contours_filtered'), contours=contours, contour_thickness=2)

#     ## Done! -> contours
#     return contours


# def bf_laplacian(f, dinfo: DInfo):
    
#     l = CellSegmentMods.laplacian_of_gaussian(f, sigma=2, ksize=7)

#     ### Colony outline from laplacian -> m0
#     # for t in [1,2,3,4,5,10]:
#     #     for s in [-1,-2,-3, -4, -5, -10]:
#     #         M00 =  np.logical_or(s > M0, M0 > t)
#     #         MKSegmentUtils.plot_frame(M00, title=f'{label}_col_2_M00_{s}_{t}', plot=plot, dir=debug_dir, crop=crop)

#     # MKSegmentUtils.plot_frame(-THRESHOLD >= m0, dinfo=dinfo.append_to_label('2_m0n'))
#     # MKSegmentUtils.plot_frame(THRESHOLD <= m0, dinfo=dinfo.append_to_label('2_m0p'))

#     # m0 =  np.logical_or(-THRESHOLD >= m0, m0 >= THRESHOLD)
#     m0 = 1000 <= l
#     MKSegmentUtils.plot_frame(m0, dinfo=dinfo.append_to_label('2_m0'))


#     # moiph fill
#     m0 = m0.astype(np.uint8)
#     # remove background noise
#     m0 = cv.morphologyEx(m0, cv.MORPH_OPEN, CellSegmentMods.kc3, iterations=1)
#     MKSegmentUtils.plot_frame(MKSegmentUtils.norm(m0), dinfo=dinfo.append_to_label('c1'))
#     # fill holes
#     m0 = cv.morphologyEx(m0, cv.MORPH_CLOSE, CellSegmentMods.kc7, iterations=2)
#     MKSegmentUtils.plot_frame(MKSegmentUtils.norm(m0), dinfo=dinfo.append_to_label('c2'))
#     # m0 = cv.morphologyEx(m0, cv.MORPH_CLOSE, CellSegmentMods.kc5, iterations=2)
#     # MKSegmentUtils.plot_frame(MKSegmentUtils.norm(m0), dinfo=dinfo.append_to_label('c2'))
#     # m0 = cv.morphologyEx(m0, cv.MORPH_CLOSE, CellSegmentMods.kc5, iterations=2)
#     # MKSegmentUtils.plot_frame(MKSegmentUtils.norm(m0), dinfo=dinfo.append_to_label('c2'))
    
#     # m0 = cv.morphologyEx(m0, cv.MORPH_OPEN, CellSegmentMods.kc3, iterations=1)
#     # MKSegmentUtils.plot_frame(MKSegmentUtils.norm(m0), dinfo=dinfo.append_to_label('c1'))

    
#     # Agressive thresholding to remove debris and lysed colonies

#     m1 = 4000 <= l
#     MKSegmentUtils.plot_frame(m1, dinfo=dinfo.append_to_label('2_m1'))

#     m1 = m1.astype(np.uint8)
#     m1 = cv.morphologyEx(m1, cv.MORPH_OPEN, CellSegmentMods.kc3, iterations=1)
#     MKSegmentUtils.plot_frame(MKSegmentUtils.norm(m1), dinfo=dinfo.append_to_label('c1'))
#     m1 = cv.morphologyEx(m1, cv.MORPH_DILATE, CellSegmentMods.kc21, iterations=2)
#     MKSegmentUtils.plot_frame(MKSegmentUtils.norm(m1), dinfo=dinfo.append_to_label('c1'))


#     m0 = segmentation.clear_border(m0, buffer_size=MIN_EDGE_DISTANCE)
#     MKSegmentUtils.plot_frame(MKSegmentUtils.norm(m0), dinfo=dinfo.append_to_label('c1'))

#     # label then filter
#     # ml, num_features = ndimage.label(m0)

#     # # filter 
#     # for i in range(1, num_features+1):
#     #     mask = ml==i
        
#     #     if not CellSegmentMods.should_keep_colony(mask, filter_mask=m1):
#     #         ml = np.where(mask, 0, ml)
    
        
#     contours, _  = cv.findContours((m0).astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#     MKSegmentUtils.plot_frame(f, dinfo=dinfo.append_to_label('8_contours_all'), contours=contours, contour_thickness=2)
    
#     contours = MKSegmentUtils.contour_filter(contours, min_area=MKSegmentUtils.MIN_COLONY_AREA)
#     MKSegmentUtils.plot_frame(f, dinfo=dinfo.append_to_label('9_contours_filtered'), contours=contours, contour_thickness=2)
    
#     p = CellSegmentMods.center_of_widest_point(contours[0])
    
#     # filter to only keep contours with center 
#     contours = [c for c in contours if m1[tuple(CellSegmentMods.center_of_widest_point(c))]]

    
#     p0s = [CellSegmentMods.center_of_widest_point(c) for c in contours]    

#     ## Done! -> contours
#     return contours


# def bf_via_edges(frame, dinfo, lower_threshold=30, upper_threshold=170, min_area=MKSegmentUtils.MIN_COLONY_AREA, close_iterations=2):
    
    # # Apply Gaussian blur
    # blurred_image = gray_image # cv.GaussianBlur(gray_image, (3, 3), 0)
    
    # # Apply the Canny edge detector
    # edges = cv.Canny(blurred_image, lower_threshold, upper_threshold)

    # # # Dilate the edges to make them thicker
    # # dilated_edges = cv.dilate(edges, None, iterations=3)  # Increase the iterations if necessary
    
    # # # Erode to bring them back to a similar width but ensure thickness
    # # final_edges = cv.erode(dilated_edges, None, iterations=3)  # Match the iterations used in dilation
    # final_edges = edges

    # return final_edges

#     col_edges = CellSegmentMods.robust_edge_detection(frame, lower_threshold=lower_threshold, upper_threshold=upper_threshold)
#     MKSegmentUtils.plot_frame(MKSegmentUtils.norm(col_edges), dinfo=dinfo.append_to_label('1_col_edges'))
    
#     col_edges = segmentation.clear_border(col_edges, buffer_size=MIN_EDGE_DISTANCE)
#     MKSegmentUtils.plot_frame(MKSegmentUtils.norm(col_edges), dinfo=dinfo.append_to_label('2_clear_from_border'))

#     # moiph fill
#     moiph = cv.morphologyEx(col_edges, cv.MORPH_CLOSE, CellSegmentMods.kc5, iterations=close_iterations)
#     MKSegmentUtils.plot_frame(MKSegmentUtils.norm(moiph), dinfo=dinfo.append_to_label('3_col_edges2'))

#     contours, _  = cv.findContours(moiph.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#     MKSegmentUtils.plot_frame(frame, dinfo=dinfo.append_to_label('4_contours_all'), contours=contours, contour_thickness=2)
    
#     contours = MKSegmentUtils.contour_filter(contours, min_area=min_area)
#     MKSegmentUtils.plot_frame(frame, dinfo=dinfo.append_to_label('5_contours_filtered'), contours=contours, contour_thickness=2)

#     ## Done! -> contours
#     return contours

from skimage import feature

# min_mean_contrast -1000 found in staph_tet_BE224_20_C2_bf_i13_t07241_11246-11257-l12

def bf_via_edges(frame, dinfo, min_area=MKSegmentUtils.MIN_COLONY_AREA, close_iterations=2, min_mean_contrast=-1050, max_mean_contrast=100, min_negative_contrast=200):
    
    col_edges = feature.canny(frame, sigma=1).astype(np.uint8)
    MKSegmentUtils.plot_frame(MKSegmentUtils.norm(col_edges), dinfo=dinfo.append_to_label('1_col_edges'))
    
    # moiph fill
    masks = cv.morphologyEx(col_edges, cv.MORPH_CLOSE, CellSegmentMods.kc7, iterations=close_iterations)
    MKSegmentUtils.plot_frame(MKSegmentUtils.norm(masks), dinfo=dinfo.append_to_label('3_close'))
 
    masks = segmentation.clear_border(masks, buffer_size=MIN_EDGE_DISTANCE)
    MKSegmentUtils.plot_frame(MKSegmentUtils.norm(col_edges), dinfo=dinfo.append_to_label('2_clear_from_border'))

    contours, _  = cv.findContours(masks.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    MKSegmentUtils.plot_frame(frame, dinfo=dinfo.append_to_label('4_contours_all'), contours=contours, contour_thickness=2)
    
    contours = [c for c in contours if MKSegmentUtils.contour_to_area(c) > min_area]
    MKSegmentUtils.plot_frame(frame, dinfo=dinfo.append_to_label('5_area_filtered'), contours=contours, contour_thickness=2)
    
    # filter colonies that have low contrast -> lysed
    laplaican = CellSegmentMods.laplacian_of_gaussian(frame, sigma=2, ksize=7) # laplacian of gaussian invert so bright spots are positive
    contours = filter_contours_by_laplacian_profile(laplacian=laplaican, contours=contours, dinfo=dinfo.with_file_plot())

    # compute average value of l within each contour
    # avg_l_both = [CellSegmentMods.mean_value_in_contour(c, laplaican) for c in contours] 
    # avg_l_neg = [CellSegmentMods.mean_value_in_contour(c, laplaican_neg) for c in contours]
    # std_l_both = [CellSegmentMods.std_value_in_contour(c, laplaican) for c in contours]

    # avg_l_both_str = [f'{l:.1f}' for l in avg_l_both]
    # avg_l_neg_str = [f'{l:.1f}' for l in avg_l_neg]
    # std_l_both_str = [f'{l:.1f}' for l in std_l_both]
    # MKSegmentUtils.plot_frame(frame, dinfo=dinfo.append_to_label('6_contrast_label_both'), contours=contours, contour_thickness=2, contour_labels=avg_l_both_str)
    # MKSegmentUtils.plot_frame(frame, dinfo=dinfo.append_to_label('6_contrast_label_neg'), contours=contours, contour_thickness=2, contour_labels=avg_l_neg_str)
    # MKSegmentUtils.plot_frame(frame, dinfo=dinfo.append_to_label('6_contrast_label_std_both'), contours=contours, contour_thickness=2, contour_labels=std_l_both_str)
    #  print(avg_l_both_str)
    # print(avg_l_neg_str)

    # contours = [c for c, l in zip(contours, avg_l_both) if min_mean_contrast < l < max_mean_contrast]
    # contours = [c for c, l in zip(contours, avg_l_neg) if min_negative_contrast < l]
    # MKSegmentUtils.plot_frame(frame, dinfo=dinfo.append_to_label('6_contrast_filtered'), contours=contours, contour_thickness=2)
 
    # m1 = 4000 <= l
    # MKSegmentUtils.plot_frame(m1, dinfo=dinfo.append_to_label('2_m1'))
    # m1 = m1.astype(np.uint8)
    # m1 = cv.morphologyEx(m1, cv.MORPH_OPEN, CellSegmentMods.kc3, iterations=1)
    # MKSegmentUtils.plot_frame(MKSegmentUtils.norm(m1), dinfo=dinfo.append_to_label('c1'))
    # m1 = cv.morphologyEx(m1, cv.MORPH_DILATE, CellSegmentMods.kc21, iterations=2)
    # MKSegmentUtils.plot_frame(MKSegmentUtils.norm(m1), dinfo=dinfo.append_to_label('c1'))
    # contours = [c for c in contours if m1[tuple(CellSegmentMods.center_of_widest_point(c))]]
    # MKSegmentUtils.plot_frame(frame, dinfo=dinfo.append_to_label('5_contrast_filtered'), contours=contours, contour_thickness=2)

    ## Done! -> contours
    return contours