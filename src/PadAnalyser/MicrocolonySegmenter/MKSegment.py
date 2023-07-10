from .MKSegmentUtils import *
import logging

# '''
# Preporcesses frame and returns (frame, laplacian)
# '''
# def preprocess(f, dinfo):
#     ### Initial normalization -> c0
#     # normalize max intensity - if nothing in frame, all pixels will be bright as opposed to if normalizing both upper and lower bound
#     c0 = f
#     c0 = normalize_up(c0)
#     plot_frame(c0, dinfo=dinfo.append_to_label('0_start'))
#     if dinfo.printing:
#         logging.debug(f'Preprocessed frame info: type={c0.dtype}, min={np.min(c0)}, max={np.max(c0)}')

#     return c0


# def laplacian(f, dinfo):
#     ### Compute second order spatial derivatives -> l
#     l = f
#     l = cv.GaussianBlur(l, (3, 3), 0)
    
#     # Testing laplacian kernel size
#     # for s in [5,7,9,11,13,15]:
#     #     l0 = cv.Laplacian(l, cv.CV_16S, ksize=s)
#     #     plot_frame(norm(l0), title=f'{label}_bf_1_laplacian_{s}', plot=plot, dir=debug_dir, crop=crop)
    
#     l = cv.Laplacian(l, cv.CV_16S, ksize=7)
#     l = l//256

#     plot_frame(norm(l), dinfo=dinfo.append_to_label('1_l'))
#     if dinfo.printing:
#         logging.debug(f'Laplacian frame info: min={np.min(l)}, max={np.max(l)}')

#     return l

score_kernel = np.array([
    [1,1,1],
    [1,2,1],
    [1,1,1],
])

def closest_factor(n,k):    
    for i in range(int(k*0.9),n):
        if n%i == 0:
            return i

def focus_stitching(stack):
    if type(stack) != list:
        print('Stack has only one frame, returning that')
        return stack 

    fs = stack
    fs = [cv.GaussianBlur(f, (7, 7), 0) for f in fs] # blur, kernel size about feature size
    fs = [cv.Laplacian(f, cv.CV_32S, ksize=7) for f in fs] # laplacian

    fs_abs = [np.square(f) for f in fs] # square to make all positive and emphasize large values over larger area with smaller amplitude
    
    # for f in fs_abs:
    #     print(np.max(f), np.min(f))

    # size-representative frame
    height, width = stack[0].shape

    x_splits = list(range(0, width+1, closest_factor(width, WINDOW_SIZE)))
    y_splits = list(range(0, height+1, closest_factor(height, WINDOW_SIZE)))

    x_ranges = [slice(a,b) for a,b in zip(x_splits, x_splits[1:])]
    y_ranges = [slice(a,b) for a,b in zip(y_splits, y_splits[1:])]

    weighted_scores = [np.array([[np.mean(l[ys,xs]) for ys in y_ranges] for xs in x_ranges]) for l in fs_abs]
    weighted_scores = [scipy.signal.convolve2d(d, score_kernel, mode='same') for d in weighted_scores]

    f = np.zeros_like(stack[0]).astype(np.float32) # new frame to build and return
    for i, xs in enumerate(x_ranges):
        for j, ys in enumerate(y_ranges):
            scores = [l[i,j] for l in weighted_scores]
            focus_index = scores.index(max(scores))
            
            f = f + window_kernel(xs, ys, f.shape) * stack[focus_index] # add this window to return frame

    return f.astype(stack[0].dtype)


# simple flatten stack with stitching 
def flatten_stack(stack, dinfo):
    frame_raw = normalize_up(focus_stitching(stack)) # compute laplacian from normalized frame
    frame = to_dtype_uint8(frame_raw)

    # compute laplacian compressed stack
    laplacian_frame = cv.GaussianBlur(frame_raw, (7, 7), 0) # blur, kernel size about feature size
    laplacian_frame = cv.Laplacian(laplacian_frame, cv.CV_16S, ksize=7) # laplacian
    laplacian_frame = laplacian_frame//2**8 # scale to fit in int16
    laplacian_frame = laplacian_frame.astype(np.int16)

    # output debug frames
    plot_frame(frame, dinfo=dinfo.append_to_label('z_stack_best'))
    plot_frame(laplacian_frame, dinfo=dinfo.append_to_label(f'z_stack_laplacian'))
    # for i, s in enumerate(stack):
    #     plot_frame(s, dinfo=dinfo.append_to_label(f'z_stack_frame_{i}'))

    return frame, laplacian_frame

# # simple flatten stack 
# def flatten_stack(stack, dinfo):
#     fs = stack
#     fs = [cv.GaussianBlur(f, (7, 7), 0) for f in fs] # blur, kernel size about feature size
#     fs = [cv.Laplacian(f, cv.CV_32S, ksize=7) for f in fs] # laplacian

#     # find best normal frame
#     fs_abs = [np.absolute(f) for f in fs] # absolute value
#     scores = [np.mean(l) for l in fs_abs] # compute score
#     focus_index = scores.index(max(scores))
#     frame = norm_up(stack[focus_index])

#     # compute laplacian compressed stack
#     # laplacian_frame = np.mean(np.array(fs), 0) # option 1: compute laplacian as mean of all frames
#     laplacian_frame = fs[focus_index] # option 2: pick frame with best focus

#     laplacian_frame = laplacian_frame//2**(16 if stack[0].dtype == np.uint16 else 8)
#     laplacian_frame = laplacian_frame.astype(np.int16)

#     # output debug frames
#     plot_frame(frame, dinfo=dinfo.append_to_label('z_stack_best'))
#     plot_frame(laplacian_frame, dinfo=dinfo.append_to_label(f'z_stack_laplacian'))
#     for i, s in enumerate(stack):
#         plot_frame(s, dinfo=dinfo.append_to_label(f'z_stack_frame_{i}'))

#     return frame, laplacian_frame, focus_index



'''Assume frame has been preprocessed, and is uint16'''
def bf_single_cell_segment(f, colony_masks, dinfo):

    if dinfo.printing:
        logging.debug(f'Raw laplacian frame info: type={l.dtype}, min={np.min(l)}, max={np.max(l)}')

    ### Make single cell mask -> m0
    # m0 = l
    # for s in [-4,-5,-6, -7, -8, -9, -10, -11, -12, -13, -14, -15]:
    #     m00 =  m0 < s
    #     plot_frame(m00, dinfo=dinfo.append_to_label(f'00_{s}'))
    
    # m0 =  m0 < -4 # used to be -8, made smaller to get less fragmented masks
    # plot_frame(m0, dinfo=dinfo.append_to_label('2_m0'))

    ### Include intensity from BF to get rid of cell outlines -> m1
    m1 = f
    m1 = cv.GaussianBlur(m1, (3, 3), 0)
    # for o in [0,-1,-2,-3,-4,-5,-6,-7,-8,-9]:
    #     m10 = cv.adaptiveThreshold(m1, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 41, o)
    #     plot_frame(m10, dinfo=dinfo.append_to_label(f'3_m1_{o}'))
    # for o in [11,21,31,41,51,61,71,81,91,101]:
    #     m10 = cv.adaptiveThreshold(m1, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, o, -4)
    #     plot_frame(m10, dinfo=dinfo.append_to_label(f'3_m1__{o}'))

    m1 = cv.adaptiveThreshold(m1, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 81, -4)
    plot_frame(m1, dinfo=dinfo.append_to_label('3_m11'))
    
    # m1 = cv.morphologyEx(m1, cv.MORPH_OPEN, k5_circle)
    # plot_frame(m1, dinfo=dinfo.append_to_label('3_m12'))

    # m1 = np.logical_and(m1, scipy.signal.convolve2d(m1, k5_circle, mode='same') < 5) # fill larger holes

    # anything that fits inside an nxn square of 1s is removed
    
    # filter anything outside colony mask
    cm = np.zeros(f.shape).astype(np.uint8)
    for i, _ in enumerate(colony_masks):
        cv.drawContours(cm, contours=colony_masks, contourIdx=i, color=1, thickness=cv.FILLED)
    plot_frame(cm*255, dinfo=dinfo.append_to_label('3_m2'))
    
    m1 = np.logical_and(m1, cm)

    # # m1 = cv.morphologyEx(m1, cv.MORPH_OPEN, k2_square)
    # plot_frame(m1, dinfo=dinfo.append_to_label('3_m3'))
    # # m1 = cv.morphologyEx(m1, cv.MORPH_OPEN, k9_circle)
    # # plot_frame(m1, dinfo=dinfo.append_to_label('4_a'))
    # # m1 = cv.morphologyEx(m1, cv.MORPH_CLOSE, kernel)
    # # plot_frame(m1, dinfo=dinfo.append_to_label('4_b'))


    # ### Combine masks -> m
    # m = m1 # np.logical_and(m0, m1)
    # plot_frame(m, dinfo=dinfo.append_to_label('4_m'))

    ### Fill holes -> mu
    # mu = m
    # mu0 = scipy.signal.convolve2d(mu, k6_square, mode='same') >= 15 # fill larger holes
    # mu = np.logical_or(mu, mu0)
    # mu0 = scipy.signal.convolve2d(mu, k4_square, mode='same') >= 8 # fill small remaining holes
    # mu = np.logical_or(mu, mu0)
    # plot_frame(mu, dinfo=dinfo.append_to_label('5_mu'))
    
    # # Could be  effective: 

    # ### Filter any elements smaller than n blocks -> mf
    # mf = mu
    # mf = scipy.signal.convolve2d(mf, k5, mode='same') > 4
    # mf = np.logical_and(mf, m)
    # plot_frame(mf, dinfo=dinfo.append_to_label('6_mf'))

    ### Detect contours -> contours
    mf = m1
    contours, hierarchy = cv.findContours(np.uint8(mf), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    plot_frame(f, dinfo=dinfo.append_to_label('07_contours_all'), contours=contours)

    contours = contour_filter(contours, hierarchy, min_area=MIN_CELL_AREA)
    plot_frame(f, dinfo=dinfo.append_to_label('08_contours_filtered'), contours=contours, contour_thickness=cv.FILLED)
    
    contours = [c_out for c_in in contours for c_out in split_contour_by_point_distance(c_in, preview=False)]
    plot_frame(f, dinfo=dinfo.append_to_label('09_contours_split_point_distance'), contours=contours, contour_thickness=cv.FILLED)
    
    contours = [c_out for c_in in contours for c_out in split_contour_by_curvature(c_in, preview=False)]
    plot_frame(f, dinfo=dinfo.append_to_label('10_contours_split_curvature'), contours=contours, contour_thickness=cv.FILLED)
    
    contours = [dilate_contour(c) for c in contours]
    plot_frame(f, dinfo=dinfo.append_to_label('11_contours_dilated'), contours=contours, contour_thickness=cv.FILLED)

    ### Done! -> contours
    return contours



# TODO: delete
# def colony_contours(l, dinfo):
    
#     m0 = l
#     m0 =  np.logical_or(-5 >= m0, m0 >= 5)

#     ### Close borders -> m2
#     m2 = m0
#     m2 = scipy.signal.convolve2d(m2, k9_circle, mode='same') > np.sum(k9_circle)*1/2
#     m2 = np.logical_or(m2, m0)
#     plot_frame(m2, dinfo=dinfo.append_to_label('4_m2'))

#     # ### Filter any elements smaller than n blocks -> m1
#     m1 = m2
#     m1 = scipy.signal.convolve2d(m1, kL1, mode='same') > np.sum(kL1)//2
#     m1 = np.logical_and(m1, m2)
#     plot_frame(m1, dinfo=dinfo.append_to_label('3_m1'))
    
#     ### Fill holes -> mu
#     mu = m1
#     mu = cv.morphologyEx(mu.astype(np.uint8), cv.MORPH_CLOSE, kL2.astype(np.uint8)) # fill last remaining holes
#     plot_frame(mu.astype(bool), dinfo=dinfo.append_to_label('5_mu'))
#     mu = cv.morphologyEx(mu.astype(np.uint8), cv.MORPH_OPEN, kL2.astype(np.uint8)) # harsh debris removal
#     plot_frame(mu.astype(bool), dinfo=dinfo.append_to_label('6_mu'))
#     mu = cv.GaussianBlur(mu, (41, 41), 0)
#     # mu = cv.morphologyEx(mu.astype(np.uint8), cv.MORPH_DILATE, k5_circle.astype(np.uint8)) # make mask a bit bigger
#     plot_frame(mu.astype(bool), dinfo=dinfo.append_to_label('7_mu'))

#     return mu


def bf_colony_segment(l, dinfo):
        
    ### Colony outline from laplacian -> m0
    m0 = l
    # for t in [1,2,3,4,5,10]:
    #     for s in [-1,-2,-3, -4, -5, -10]:
    #         M00 =  np.logical_or(s > M0, M0 > t)
    #         plot_frame(M00, title=f'{label}_col_2_M00_{s}_{t}', plot=plot, dir=debug_dir, crop=crop)

    m0 =  np.logical_or(-5 >= m0, m0 >= 5)
    # m0 =  m0 < -8
    plot_frame(m0, dinfo=dinfo.append_to_label('2_m0'))

    ### Close borders -> m2
    m2 = m0
    m2 = scipy.signal.convolve2d(m2, k9_circle, mode='same') > np.sum(k9_circle)*1/2
    m2 = np.logical_or(m2, m0)
    plot_frame(m2, dinfo=dinfo.append_to_label('4_m2'))

    # ### Filter any elements smaller than n blocks -> m1
    m1 = m2
    m1 = scipy.signal.convolve2d(m1, kL1, mode='same') > np.sum(kL1)//2
    m1 = np.logical_and(m1, m2)
    plot_frame(m1, dinfo=dinfo.append_to_label('3_m1'))
    
    ### Fill holes -> mu
    mu = m1
    mu = cv.morphologyEx(mu.astype(np.uint8), cv.MORPH_CLOSE, kL2.astype(np.uint8)) # fill last remaining holes
    plot_frame(mu.astype(bool), dinfo=dinfo.append_to_label('5_mu'))
    mu = cv.morphologyEx(mu.astype(np.uint8), cv.MORPH_OPEN, kL2.astype(np.uint8)) # harsh debris removal
    plot_frame(mu.astype(bool), dinfo=dinfo.append_to_label('6_mu'))
    mu = cv.GaussianBlur(mu, (41, 41), 0)
    # mu = cv.morphologyEx(mu.astype(np.uint8), cv.MORPH_DILATE, k5_circle.astype(np.uint8)) # make mask a bit bigger
    plot_frame(mu.astype(bool), dinfo=dinfo.append_to_label('7_mu'))


    ### Contour detection -> ca
    contours, hierarchy = cv.findContours(np.uint8(mu), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    plot_frame(l, dinfo=dinfo.append_to_label('8_contours_all'), contours=contours, contour_thickness=2)
    
    contours = contour_filter(contours, hierarchy, min_area=MIN_COLONY_AREA)
    plot_frame(l, dinfo=dinfo.append_to_label('9_contours_filtered'), contours=contours, contour_thickness=2)

    ### Done! -> contours
    return contours





# def test():
#     debug_dir = '/Users/mkals/Desktop/BE39/debug'
#     MKUtils.generate_directory(debug_dir)

#     start_idx = 0
#     for label, f in zip(filenames[start_idx:], frames[start_idx:start_idx+1]):
#         bf_colony_cell_segment(f, 0, f'test_seg_{label}', plot=False, debug_dir=debug_dir)#, crop=((600, 1000),(600, 1000)))


# if __name__ == '__main__':
#     test()