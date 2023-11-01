
from . import MKSegmentUtils, DInfo, CellSegmentMods
import numpy as np
import cv2 as cv
import scipy.signal
from skimage import segmentation


def bf_colony_segment(f, dinfo: DInfo):
    
    l = CellSegmentMods.laplacian_uint8(f)

    ### Colony outline from laplacian -> m0
    m0 = l
    # for t in [1,2,3,4,5,10]:
    #     for s in [-1,-2,-3, -4, -5, -10]:
    #         M00 =  np.logical_or(s > M0, M0 > t)
    #         MKSegmentUtils.plot_frame(M00, title=f'{label}_col_2_M00_{s}_{t}', plot=plot, dir=debug_dir, crop=crop)

    THRESHOLD = 2000
    m0 =  np.logical_or(-THRESHOLD >= m0, m0 >= THRESHOLD)
    # m0 =  m0 < -8
    MKSegmentUtils.plot_frame(m0, dinfo=dinfo.append_to_label('2_m0'))

    ### Close borders -> m2
    m2 = m0
    m2 = scipy.signal.convolve2d(m2, MKSegmentUtils.k9_circle, mode='same') > np.sum(MKSegmentUtils.k9_circle)*1/2
    m2 = np.logical_or(m2, m0)
    MKSegmentUtils.plot_frame(m2, dinfo=dinfo.append_to_label('3_m2'))

    # ### Filter any elements smaller than n blocks -> m1
    m1 = m2
    m1 = scipy.signal.convolve2d(m1, MKSegmentUtils.kL1, mode='same') > np.sum(MKSegmentUtils.kL1)//2
    m1 = np.logical_and(m1, m2)
    MKSegmentUtils.plot_frame(m1, dinfo=dinfo.append_to_label('4_m1'))
    
    ### Fill holes -> mu
    mu = m1
    mu = cv.morphologyEx(mu.astype(np.uint8), cv.MORPH_CLOSE, MKSegmentUtils.kL2.astype(np.uint8)) # fill last remaining holes
    MKSegmentUtils.plot_frame(mu.astype(bool), dinfo=dinfo.append_to_label('5_mu'))
    mu = cv.morphologyEx(mu.astype(np.uint8), cv.MORPH_OPEN, MKSegmentUtils.kL2.astype(np.uint8)) # harsh debris removal
    MKSegmentUtils.plot_frame(mu.astype(bool), dinfo=dinfo.append_to_label('6_mu'))
    mu = cv.GaussianBlur(mu, (21, 21), 0)
    # mu = cv.morphologyEx(mu.astype(np.uint8), cv.MORPH_DILATE, k5_circle.astype(np.uint8)) # make mask a bit bigger
    MKSegmentUtils.plot_frame(mu.astype(bool), dinfo=dinfo.append_to_label('7_mu'))

    mu = segmentation.clear_border(mu)

    ### Contour detection -> ca
    # contours = label_contour_in_mask(mu, dinfo=dinfo.append_to_label('8_lbl'))
    # MKSegmentUtils.plot_frame(l, dinfo=dinfo.append_to_label('8_contours_all'), contours=contours, contour_thickness=2)

    # return contours

    # filter regions touching border
    


    contours, _  = cv.findContours(np.uint8(mu), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    MKSegmentUtils.plot_frame(l, dinfo=dinfo.append_to_label('8_contours_all'), contours=contours, contour_thickness=2)
    
    contours = MKSegmentUtils.contour_filter(contours, min_area=MKSegmentUtils.MIN_COLONY_AREA)
    MKSegmentUtils.plot_frame(l, dinfo=dinfo.append_to_label('9_contours_filtered'), contours=contours, contour_thickness=2)

    ## Done! -> contours
    return contours


def bf_via_edges(frame, dinfo):
    
    col_edges = CellSegmentMods.robust_edge_detection(frame, lower_threshold=30, upper_threshold=170)
    MKSegmentUtils.plot_frame(MKSegmentUtils.norm(col_edges), dinfo=dinfo.append_to_label('col_edges'))
    
    # moiph fill
    moiph = cv.morphologyEx(col_edges, cv.MORPH_CLOSE, CellSegmentMods.kc5, iterations=2)
    MKSegmentUtils.plot_frame(MKSegmentUtils.norm(moiph), dinfo=dinfo.append_to_label('col_edges2'))
    

    contours, _  = cv.findContours(moiph.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    MKSegmentUtils.plot_frame(frame, dinfo=dinfo.append_to_label('8_contours_all'), contours=contours, contour_thickness=2)
    
    contours = MKSegmentUtils.contour_filter(contours, min_area=MKSegmentUtils.MIN_COLONY_AREA)
    MKSegmentUtils.plot_frame(frame, dinfo=dinfo.append_to_label('9_contours_filtered'), contours=contours, contour_thickness=2)

    ## Done! -> contours
    return contours