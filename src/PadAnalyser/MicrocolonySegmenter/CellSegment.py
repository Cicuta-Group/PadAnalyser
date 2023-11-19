from . import MKSegmentUtils, DInfo, CellSegmentMods
import numpy as np
import cv2 as cv

import logging
from scipy.ndimage import convolve


'''

Group of methods that area all designed for segmenting single cells. 
Some may work better in some scenarios than others, and vice versa. 

'''


'''
Assume frame has been preprocessed, and is uint8
Split cells using watershed method

Good at segmenting spherical morphologies. 
Issue: arbitrary limit between looking for small and large cells in label method. 
'''
def bf_watershed(f, colony_contours, dinfo: DInfo):

    # make frame from colony contours
    colony_mask = np.zeros_like(f, dtype=np.uint8)
    cv.drawContours(colony_mask, colony_contours, -1, 1, cv.FILLED)

    if dinfo.printing:
        logging.debug(f'Raw frame info: type={f.dtype}, min={np.min(f)}, max={np.max(f)}')

    ### Include intensity from BF to get rid of cell outlines -> m1
    m1 = f
    m1 = cv.GaussianBlur(m1, (3, 3), 0)
    m1 = cv.adaptiveThreshold(m1, 1, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 81, -4) # -4, relativley low, to make better at segmenting big cells such as Mecilinam-exposed ecoli
    MKSegmentUtils.plot_frame(m1*255, dinfo=dinfo.append_to_label('3_m1'))

    # remove background based on colony locations
    m1 = m1 * colony_mask

    # remove background lonely pixels 
    kernel = np.ones((5, 5))
    isolated_pixels = convolve(m1, kernel) < 10
    m1 = m1 & ~isolated_pixels

    MKSegmentUtils.plot_frame(m1*255, dinfo=dinfo.append_to_label('3_m2'))

    ## Detect contours -> contours
    contours = CellSegmentMods.label_contour_in_mask(m1, dinfo=dinfo.append_to_label('4_lbl'))
    MKSegmentUtils.plot_frame(f, dinfo=dinfo.append_to_label('07_contours_all'), contours=contours, contour_thickness=cv.FILLED)

    contours = [CellSegmentMods.dilate_contour(c) for c in contours]
    MKSegmentUtils.plot_frame(f, dinfo=dinfo.append_to_label('11_contours_dilated'), contours=contours, contour_thickness=cv.FILLED)

    return contours




'''
Original segmentation method used on 100 series datasets.
Assume frame has been preprocessed, and is uint8

Issues: struggles with merging masks for close and small spherical cells.
'''
def bf_contour_operations(f, colony_contours, dinfo: DInfo):

    MKSegmentUtils.plot_frame(f, dinfo=dinfo.append_to_label('0_f'))

    if dinfo.printing:
        logging.debug(f'Raw frame info: type={f.dtype}, min={np.min(f)}, max={np.max(f)}')

    ### Include intensity from BF to get rid of cell outlines -> m1
    m1 = f
    # m1 = cv.GaussianBlur(m1, (3, 3), 0)
    m1 = cv.adaptiveThreshold(m1, 1, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 41, -6) # -4, relativley low, to make better at segmenting big cells such as Mecilinam-exposed ecoli
    MKSegmentUtils.plot_frame(m1*255, dinfo=dinfo.append_to_label('3_m1'))

    # remove background based on colony locations
    CellSegmentMods.filter_to_colonies(m1, colony_contours)
    MKSegmentUtils.plot_frame(m1*255, dinfo=dinfo.append_to_label('3_m2'))

    # remove background lonely pixels 
    kernel = np.ones((5, 5))
    isolated_pixels = convolve(m1, kernel) < 10
    m1 = m1 & ~isolated_pixels
    MKSegmentUtils.plot_frame(m1*255, dinfo=dinfo.append_to_label('3_m11'))

    mf = m1
    # contours, hierarchy = cv.findContours(np.uint8(mf), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv.findContours(np.uint8(mf), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    MKSegmentUtils.plot_frame(f, dinfo=dinfo.append_to_label('07_contours_all'), contours=contours)

    contours = MKSegmentUtils.contour_filter(contours, hierarchy, min_area=MKSegmentUtils.MIN_CELL_AREA)
    MKSegmentUtils.plot_frame(f, dinfo=dinfo.append_to_label('08_contours_filtered'), contours=contours, contour_thickness=cv.FILLED)    

    contours = [c_out for c_in in contours for c_out in MKSegmentUtils.split_contour_by_point_distance(c_in, preview=False)]
    MKSegmentUtils.plot_frame(f, dinfo=dinfo.append_to_label('09_contours_split_point_distance'), contours=contours, contour_thickness=cv.FILLED)
    
    contours = [c_out for c_in in contours for c_out in MKSegmentUtils.split_contour_by_curvature(c_in, preview=False)]
    MKSegmentUtils.plot_frame(f, dinfo=dinfo.append_to_label('10_contours_split_curvature'), contours=contours, contour_thickness=cv.FILLED)
    
    contours = [CellSegmentMods.dilate_contour(c) for c in contours]
    MKSegmentUtils.plot_frame(f, dinfo=dinfo.append_to_label('11_contours_dilated'), contours=contours, contour_thickness=cv.FILLED)

    return contours



'''
Updated segmentation with laplacian again. 
Assume frame has been preprocessed, and is uint8
'''
def bf_laplacian(frame, colony_contours, dinfo: DInfo, sigma=1, ksize=7, threshold=-2000, split_factor=0.1, min_mask_size_filter=4) -> list[np.array]:

    log = CellSegmentMods.laplacian_of_gaussian(frame, sigma=sigma, ksize=ksize)
    MKSegmentUtils.plot_frame(log, dinfo=dinfo.append_to_label('00_log'))
    m1 = log < threshold
    MKSegmentUtils.plot_frame(m1, dinfo=dinfo.append_to_label('01_m1'))
    m2 = CellSegmentMods.filter_to_colonies(m1, colony_contours)
    MKSegmentUtils.plot_frame(MKSegmentUtils.norm(m2), dinfo=dinfo.append_to_label('02_m2'))

    # morphological open to remove tiny objects
    # m2 = cv.morphologyEx(m2.astype(np.uint8), cv.MORPH_OPEN, CellSegmentMods.k3)
    # MKSegmentUtils.plot_frame(MKSegmentUtils.norm(m2), dinfo=dinfo.append_to_label('02_m2_morph_open'))

    contours, _  = cv.findContours(m2.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    MKSegmentUtils.plot_frame(frame, dinfo=dinfo.append_to_label('05_contours_all'), contours=contours, contour_thickness=cv.FILLED)
    
    # remove contours that are too small
    contours = [c for c in contours if cv.contourArea(c) >= min_mask_size_filter] # areas are small at this point, before dilation
    MKSegmentUtils.plot_frame(frame, dinfo=dinfo.append_to_label('06_contours_filtered_area'), contours=contours, contour_thickness=2)

    contours = [c_out for c_in in contours for c_out in MKSegmentUtils.split_contour_by_curvature(c_in, split_factor=split_factor, debug=dinfo.live_plot, printing=dinfo.printing)]
    MKSegmentUtils.plot_frame(frame, dinfo=dinfo.append_to_label('07_contours_split_curvature'), contours=contours, contour_thickness=cv.FILLED)

    contours = [CellSegmentMods.dilate_contour(c) for c in contours]
    MKSegmentUtils.plot_frame(frame, dinfo=dinfo.append_to_label('08_contours_dilated'), contours=contours, contour_thickness=cv.FILLED)

    return contours




    # MKSegmentUtils.plot_frame(f, dinfo=dinfo.append_to_label('0_f'))

    # l = CellSegmentMods.laplacian_uint8(f)
    # lnorm = MKSegmentUtils.norm(l)
    # MKSegmentUtils.plot_frame(l, dinfo=dinfo.append_to_label('1_l'))
    # MKSegmentUtils.plot_frame(lnorm, dinfo=dinfo.append_to_label('1_ln'))
    # print(f'laplacian: min={np.min(l)}, max={np.max(l)}')

    # for t in [-800,-900,-1000,-1200, -1400,-1600, -2000]:
    #     MKSegmentUtils.plot_frame(l < t, dinfo=dinfo.append_to_label(f'1_{t}'))

    # if dinfo.printing:
    #     logging.debug(f'Raw frame info: type={f.dtype}, min={np.min(f)}, max={np.max(f)}')

    # ### Include intensity from BF to get rid of cell outlines -> m1
    # m1 = f
    # # m1 = cv.GaussianBlur(m1, (3, 3), 0)
    # m1 = cv.adaptiveThreshold(lnorm, 1, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 41, 4) # -4, relativley low, to make better at segmenting big cells such as Mecilinam-exposed ecoli
    # MKSegmentUtils.plot_frame(m1*255, dinfo=dinfo.append_to_label('3_m1'))

    # # remove background based on colony locations
    # colony_mask = np.zeros_like(f, dtype=np.uint8)
    # cv.drawContours(colony_mask, colony_contours, -1, 1, cv.FILLED)
    # m1 = m1 * colony_mask
    # MKSegmentUtils.plot_frame(m1*255, dinfo=dinfo.append_to_label('3_m2'))

    # # remove background lonely pixels 
    # kernel = np.ones((5, 5))
    # isolated_pixels = convolve(m1, kernel) < 10
    # m1 = m1 & ~isolated_pixels
    # MKSegmentUtils.plot_frame(m1*255, dinfo=dinfo.append_to_label('3_m11'))

    # mf = m1
    # # contours, hierarchy = cv.findContours(np.uint8(mf), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # contours, hierarchy = cv.findContours(np.uint8(mf), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # MKSegmentUtils.plot_frame(f, dinfo=dinfo.append_to_label('07_contours_all'), contours=contours)

    # contours = MKSegmentUtils.contour_filter(contours, hierarchy, min_area=MKSegmentUtils.MIN_CELL_AREA)
    # MKSegmentUtils.plot_frame(f, dinfo=dinfo.append_to_label('08_contours_filtered'), contours=contours, contour_thickness=cv.FILLED)    

    # contours = [c_out for c_in in contours for c_out in MKSegmentUtils.split_contour_by_point_distance(c_in, preview=False)]
    # MKSegmentUtils.plot_frame(f, dinfo=dinfo.append_to_label('09_contours_split_point_distance'), contours=contours, contour_thickness=cv.FILLED)
    
    # contours = [c_out for c_in in contours for c_out in MKSegmentUtils.split_contour_by_curvature(c_in, preview=False)]
    # MKSegmentUtils.plot_frame(f, dinfo=dinfo.append_to_label('10_contours_split_curvature'), contours=contours, contour_thickness=cv.FILLED)
    
    # contours = [CellSegmentMods.dilate_contour(c) for c in contours]
    # MKSegmentUtils.plot_frame(f, dinfo=dinfo.append_to_label('11_contours_dilated'), contours=contours, contour_thickness=cv.FILLED)

    # return contours