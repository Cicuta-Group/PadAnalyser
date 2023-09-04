from . import MKSegmentUtils, DInfo
import numpy as np
import cv2 as cv
import logging
import skimage.transform
import scipy.signal


def z_stack_projection(stack, dinfo: DInfo.DInfo):
    fs = stack
    
    # Find second order gradients of each frame
    fs = [cv.GaussianBlur(f, (5, 5), 0) for f in fs] # blur, kernel size about feature size
    fs = [cv.Laplacian(f, cv.CV_32S, ksize=7) for f in fs] # laplacian

    # Only keep negative gradients and make them positive (corresponds to area inside cells when in focus)
    fs = [np.maximum(-f, 0) for f in fs]
    
    # Compute focus score for each pixel by downsampling with funciton that characterize information. Varience best when testing.
    KERNEL_SIZE = 101
    fs = [skimage.transform.resize(skimage.measure.block_reduce(f, (KERNEL_SIZE, KERNEL_SIZE), np.var), (f.shape)) for f in fs]
    
    # Find index with highest score in stack for each pixel and pick in focus frame based on that
    fs = np.array(fs)
    f_max = np.argmax(fs, 0)
    f_focus = np.take_along_axis(np.array(stack), f_max[None, ...], axis=0)[0]

    MKSegmentUtils.plot_frame(f_max, dinfo=dinfo.append_to_label('z_stack_indices'))
    MKSegmentUtils.plot_frame(f_focus, dinfo=dinfo.append_to_label('z_stack_best'))

    return f_focus


# simple flatten stack with stitching 
def flatten_stack(stack, dinfo):
    frame_raw = MKSegmentUtils.normalize_up(z_stack_projection(stack, dinfo=dinfo)) # compute laplacian from normalized frame
    frame = MKSegmentUtils.to_dtype_uint8(frame_raw)

    # compute laplacian compressed stack
    laplacian_frame = cv.GaussianBlur(frame_raw, (7, 7), 0) # blur, kernel size about feature size
    laplacian_frame = cv.Laplacian(laplacian_frame, cv.CV_16S, ksize=7) # laplacian
    laplacian_frame = laplacian_frame//2**8 # scale to fit in int16
    laplacian_frame = laplacian_frame.astype(np.int16)

    # output debug frames
    MKSegmentUtils.plot_frame(frame, dinfo=dinfo.append_to_label('z_stack_best'))
    MKSegmentUtils.plot_frame(laplacian_frame, dinfo=dinfo.append_to_label(f'z_stack_laplacian'))
    # for i, s in enumerate(stack):
    #     MKSegmentUtils.plot_frame(s, dinfo=dinfo.append_to_label(f'z_stack_frame_{i}'))

    return frame, laplacian_frame


'''Assume frame has been preprocessed, and is uint16'''
def bf_single_cell_segment(f, colony_masks, dinfo):

    if dinfo.printing:
        logging.debug(f'Raw laplacian frame info: type={l.dtype}, min={np.min(l)}, max={np.max(l)}')

    ### Make single cell mask -> m0
    # m0 = l
    # for s in [-4,-5,-6, -7, -8, -9, -10, -11, -12, -13, -14, -15]:
    #     m00 =  m0 < s
    #     MKSegmentUtils.plot_frame(m00, dinfo=dinfo.append_to_label(f'00_{s}'))
    
    # m0 =  m0 < -4 # used to be -8, made smaller to get less fragmented masks
    # MKSegmentUtils.plot_frame(m0, dinfo=dinfo.append_to_label('2_m0'))

    ### Include intensity from BF to get rid of cell outlines -> m1
    m1 = f
    m1 = cv.GaussianBlur(m1, (3, 3), 0)
    # for o in [0,-1,-2,-3,-4,-5,-6,-7,-8,-9]:
    #     m10 = cv.adaptiveThreshold(m1, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 41, o)
    #     MKSegmentUtils.plot_frame(m10, dinfo=dinfo.append_to_label(f'3_m1_{o}'))
    # for o in [11,21,31,41,51,61,71,81,91,101]:
    #     m10 = cv.adaptiveThreshold(m1, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, o, -4)
    #     MKSegmentUtils.plot_frame(m10, dinfo=dinfo.append_to_label(f'3_m1__{o}'))

    m1 = cv.adaptiveThreshold(m1, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 81, -4)
    MKSegmentUtils.plot_frame(m1, dinfo=dinfo.append_to_label('3_m11'))
    
    # m1 = cv.morphologyEx(m1, cv.MORPH_OPEN, k5_circle)
    # MKSegmentUtils.plot_frame(m1, dinfo=dinfo.append_to_label('3_m12'))

    # m1 = np.logical_and(m1, scipy.signal.convolve2d(m1, k5_circle, mode='same') < 5) # fill larger holes

    # anything that fits inside an nxn square of 1s is removed
    
    # filter anything outside colony mask
    cm = np.zeros(f.shape).astype(np.uint8)
    for i, _ in enumerate(colony_masks):
        cv.drawContours(cm, contours=colony_masks, contourIdx=i, color=1, thickness=cv.FILLED)
    MKSegmentUtils.plot_frame(cm*255, dinfo=dinfo.append_to_label('3_m2'))
    
    m1 = np.logical_and(m1, cm)

    # # m1 = cv.morphologyEx(m1, cv.MORPH_OPEN, k2_square)
    # MKSegmentUtils.plot_frame(m1, dinfo=dinfo.append_to_label('3_m3'))
    # # m1 = cv.morphologyEx(m1, cv.MORPH_OPEN, k9_circle)
    # # MKSegmentUtils.plot_frame(m1, dinfo=dinfo.append_to_label('4_a'))
    # # m1 = cv.morphologyEx(m1, cv.MORPH_CLOSE, kernel)
    # # MKSegmentUtils.plot_frame(m1, dinfo=dinfo.append_to_label('4_b'))


    # ### Combine masks -> m
    # m = m1 # np.logical_and(m0, m1)
    # MKSegmentUtils.plot_frame(m, dinfo=dinfo.append_to_label('4_m'))

    ### Fill holes -> mu
    # mu = m
    # mu0 = scipy.signal.convolve2d(mu, k6_square, mode='same') >= 15 # fill larger holes
    # mu = np.logical_or(mu, mu0)
    # mu0 = scipy.signal.convolve2d(mu, k4_square, mode='same') >= 8 # fill small remaining holes
    # mu = np.logical_or(mu, mu0)
    # MKSegmentUtils.plot_frame(mu, dinfo=dinfo.append_to_label('5_mu'))
    
    # # Could be  effective: 

    # ### Filter any elements smaller than n blocks -> mf
    # mf = mu
    # mf = scipy.signal.convolve2d(mf, k5, mode='same') > 4
    # mf = np.logical_and(mf, m)
    # MKSegmentUtils.plot_frame(mf, dinfo=dinfo.append_to_label('6_mf'))

    ### Detect contours -> contours
    mf = m1
    contours, hierarchy = cv.findContours(np.uint8(mf), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    MKSegmentUtils.plot_frame(f, dinfo=dinfo.append_to_label('07_contours_all'), contours=contours)

    contours = MKSegmentUtils.contour_filter(contours, hierarchy, min_area=MKSegmentUtils.MIN_CELL_AREA)
    MKSegmentUtils.plot_frame(f, dinfo=dinfo.append_to_label('08_contours_filtered'), contours=contours, contour_thickness=cv.FILLED)
    
    contours = [c_out for c_in in contours for c_out in MKSegmentUtils.split_contour_by_point_distance(c_in, preview=False)]
    MKSegmentUtils.plot_frame(f, dinfo=dinfo.append_to_label('09_contours_split_point_distance'), contours=contours, contour_thickness=cv.FILLED)
    
    contours = [c_out for c_in in contours for c_out in MKSegmentUtils.split_contour_by_curvature(c_in, preview=False)]
    MKSegmentUtils.plot_frame(f, dinfo=dinfo.append_to_label('10_contours_split_curvature'), contours=contours, contour_thickness=cv.FILLED)
    
    contours = [MKSegmentUtils.dilate_contour(c) for c in contours]
    MKSegmentUtils.plot_frame(f, dinfo=dinfo.append_to_label('11_contours_dilated'), contours=contours, contour_thickness=cv.FILLED)

    ### Done! -> contours
    return contours


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
    MKSegmentUtils.plot_frame(m2, dinfo=dinfo.append_to_label('4_m2'))

    # ### Filter any elements smaller than n blocks -> m1
    m1 = m2
    m1 = scipy.signal.convolve2d(m1, MKSegmentUtils.kL1, mode='same') > np.sum(MKSegmentUtils.kL1)//2
    m1 = np.logical_and(m1, m2)
    MKSegmentUtils.plot_frame(m1, dinfo=dinfo.append_to_label('3_m1'))
    
    ### Fill holes -> mu
    mu = m1
    mu = cv.morphologyEx(mu.astype(np.uint8), cv.MORPH_CLOSE, MKSegmentUtils.kL2.astype(np.uint8)) # fill last remaining holes
    MKSegmentUtils.plot_frame(mu.astype(bool), dinfo=dinfo.append_to_label('5_mu'))
    mu = cv.morphologyEx(mu.astype(np.uint8), cv.MORPH_OPEN, MKSegmentUtils.kL2.astype(np.uint8)) # harsh debris removal
    MKSegmentUtils.plot_frame(mu.astype(bool), dinfo=dinfo.append_to_label('6_mu'))
    mu = cv.GaussianBlur(mu, (41, 41), 0)
    # mu = cv.morphologyEx(mu.astype(np.uint8), cv.MORPH_DILATE, k5_circle.astype(np.uint8)) # make mask a bit bigger
    MKSegmentUtils.plot_frame(mu.astype(bool), dinfo=dinfo.append_to_label('7_mu'))


    ### Contour detection -> ca
    contours, hierarchy = cv.findContours(np.uint8(mu), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    MKSegmentUtils.plot_frame(l, dinfo=dinfo.append_to_label('8_contours_all'), contours=contours, contour_thickness=2)
    
    contours = MKSegmentUtils.contour_filter(contours, hierarchy, min_area=MKSegmentUtils.MIN_COLONY_AREA)
    MKSegmentUtils.plot_frame(l, dinfo=dinfo.append_to_label('9_contours_filtered'), contours=contours, contour_thickness=2)

    ### Done! -> contours
    return contours
