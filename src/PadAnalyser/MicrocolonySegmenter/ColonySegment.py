
from . import MKSegmentUtils, DInfo, CellSegmentMods
import numpy as np
import cv2 as cv
import scipy.signal
import scipy.ndimage as ndimage
from skimage import segmentation

MIN_EDGE_DISTANCE = 20

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
    # blurred_image = gray_image # cv2.GaussianBlur(gray_image, (3, 3), 0)
    
    # # Apply the Canny edge detector
    # edges = cv.Canny(blurred_image, lower_threshold, upper_threshold)

    # # # Dilate the edges to make them thicker
    # # dilated_edges = cv2.dilate(edges, None, iterations=3)  # Increase the iterations if necessary
    
    # # # Erode to bring them back to a similar width but ensure thickness
    # # final_edges = cv2.erode(dilated_edges, None, iterations=3)  # Match the iterations used in dilation
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
    laplaican = -CellSegmentMods.laplacian_of_gaussian(frame, sigma=2, ksize=7) # laplacian of gaussian invert so bright spots are positive
    laplaican_neg = np.maximum(laplaican, 0) 

    # compute average value of l within each contour
    avg_l_both = [CellSegmentMods.mean_value_in_contour(c, laplaican) for c in contours] 
    avg_l_neg = [CellSegmentMods.mean_value_in_contour(c, laplaican_neg) for c in contours]
    std_l_both = [CellSegmentMods.std_value_in_contour(c, laplaican) for c in contours]

    avg_l_both_str = [f'{l:.1f}' for l in avg_l_both]
    avg_l_neg_str = [f'{l:.1f}' for l in avg_l_neg]
    std_l_both_str = [f'{l:.1f}' for l in std_l_both]
    MKSegmentUtils.plot_frame(frame, dinfo=dinfo.append_to_label('6_contrast_label_both'), contours=contours, contour_thickness=2, contour_labels=avg_l_both_str)
    MKSegmentUtils.plot_frame(frame, dinfo=dinfo.append_to_label('6_contrast_label_neg'), contours=contours, contour_thickness=2, contour_labels=avg_l_neg_str)
    MKSegmentUtils.plot_frame(frame, dinfo=dinfo.append_to_label('6_contrast_label_std_both'), contours=contours, contour_thickness=2, contour_labels=std_l_both_str)
    #  print(avg_l_both_str)
    # print(avg_l_neg_str)

    contours = [c for c, l in zip(contours, avg_l_both) if min_mean_contrast < l < max_mean_contrast]
    contours = [c for c, l in zip(contours, avg_l_neg) if min_negative_contrast < l]
    MKSegmentUtils.plot_frame(frame, dinfo=dinfo.append_to_label('6_contrast_filtered'), contours=contours, contour_thickness=2)
 
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