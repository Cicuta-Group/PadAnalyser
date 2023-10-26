from . import MKSegmentUtils, DInfo, ZStack
import numpy as np
import cv2 as cv
import logging
import skimage.transform
import scipy.signal



'''Assume frame has been preprocessed, and is uint16'''
def bf_single_cell_segment(f, colony_contours, dinfo):

    # make frame from colony contours
    colony_mask = np.zeros_like(f, dtype=np.uint8)
    cv.drawContours(colony_mask, colony_contours, -1, 1, cv.FILLED)

    if dinfo.printing:
        logging.debug(f'Raw frame info: type={f.dtype}, min={np.min(f)}, max={np.max(f)}')

    ### Include intensity from BF to get rid of cell outlines -> m1
    m1 = f
    # m1 = MKSegmentUtils.norm(f) # allready done
    m1 = cv.GaussianBlur(m1, (3, 3), 0)
    # for o in [-3,-4,-5,-6,-8]:
    # for a in [11, 15, 21, 41, 61, 81]:
    #     m10 = cv.adaptiveThreshold(m1, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, a, -4)
    #     MKSegmentUtils.plot_frame(m10, dinfo=dinfo.append_to_label(f'3_m1_{a}'))

    m1 = cv.adaptiveThreshold(m1, 1, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 51, -4) # -4, relativley low, to make better at segmenting big cells such as Mecilinam-exposed ecoli
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

    m0 =  np.abs(m0) > 8

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
