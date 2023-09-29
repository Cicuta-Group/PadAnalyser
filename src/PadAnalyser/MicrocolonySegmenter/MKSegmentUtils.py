import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import logging

from PIL import Image, ImageFont, ImageDraw
import os
from colorhash import ColorHash
import scipy, scipy.signal
try:
    import polylabel_pyo3
except ImportError:
    print('Could not import polylabel_pyo3')

from scipy import ndimage


UM_PER_PIXEL = 0.112 # for Genicam with 40x objective
# UM_PER_PIXEL = 0.147 # for Grashopper with 40x objective

MIN_COLONY_AREA = 2 # in um^2
MIN_CELL_AREA = 0.25 # in um^2
MIN_POINT_SEPARATION = 4 # in um

'''
Normalize frame between 0 and 255
'''
def norm(f):
    if f.dtype == np.bool8:
        f = np.uint8(f)
    return cv.normalize(f, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)


'''
Scale max pixel value up to 255, but keep lower bound the same
'''
def normalize_up(f):
    maximum = np.max(f)
    max_possible = np.iinfo(f.dtype).max
    return (f.astype(np.float32)*max_possible/maximum).astype(f.dtype)

# def normalize_up(f0):
#     type_max = np.iinfo(f0.dtype).max
#     mean, type_mean = np.mean(f0), type_max/2
#     scaling_ratio = type_mean / mean
#     f = f0.astype(np.float32)*scaling_ratio
#     print(np.max(f), type_max)
#     f = np.maximum(f, type_max)
#     print(np.max(f))
#     return f.astype(f0.dtype)

def to_dtype_uint8(f):
    if f.dtype == np.uint8: return f
    if f.dtype == np.uint16: return (f//2**8).astype(np.uint8)
    if f.dtype == np.uint32: return (f//2**24).astype(np.uint8)
    raise TypeError(f'to_dtype_uint8() does not know how to handle data of type {f.dtype}')

# '''
# Scale max pixel value up to 255, but keep lower bound the same
# '''
# def norm_up(f):
#     minimum = np.min(f)
#     if f.dtype == np.uint16:
#         minimum = minimum // 256

#     if f.dtype == np.bool8:
#         f = np.uint8(f)
#     return cv.normalize(f, None, alpha=minimum, beta=255, norm_type=cv.NORM_MINMAX).astype(np.uint8)




##### Kernels ####


kL1 = np.ones((11,11), dtype=np.uint8)
kL2 = np.ones((7,7), dtype=np.uint8)
k4_square = np.array([
    [1, 1, 1, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 1, 1, 1],
])
k2_square = np.array([
    [1, 1],
    [1, 1],
]).astype(np.uint8)
k3_circle = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0],
]).astype(np.uint8)
k5_circle = np.array([
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
]).astype(np.uint8)
k9_circle = np.array([
    [0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0],
]).astype(np.uint8)
k6_square = np.array([
    [1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1],
])
k5 = np.ones((5,5))

kLC_size = 41
kLC = cv.getStructuringElement(cv.MORPH_ELLIPSE,(kLC_size,kLC_size)) # large circular kernel



### Contour manipulations

def dist_mask(x, k=0):
    return np.triu(x, k) - np.triu(x, x.shape[0]-k) > 0

def mask_indices(N,k):
    return np.mask_indices(N, mask_func=dist_mask, k=k)

def split_at_indices(contour, i0, i1):
    c_a = np.concatenate((contour[:i0], contour[i1:]))
    c_b = contour[i0:i1]
    return c_a, c_b

from scipy import spatial

def split_contour_by_point_distance(contour: np.array, min_distance: float = 2, preview=False):
    
    contour_reduced = contour[:,0,:]
    N = len(contour)
    k = 5
    rows, cols = mask_indices(N,k) # offset by 5 to only look at points separated by 5 points
    
    if rows.shape[0]:
        
        d_matrix = spatial.distance_matrix(x=contour_reduced, y=contour_reduced)
        # print(d_matrix, rows, cols)

        ci = np.argmin(d_matrix[rows,cols]) 
        row, col = rows[ci], cols[ci]

        if d_matrix[row, col] <= min_distance:
            ca, cb = split_at_indices(contour=contour, i0=row, i1=col) # split contour on index set (row, col)

            # check areas are sufficiently large
            if contour_to_area(ca) < MIN_CELL_AREA and contour_to_area(cb) < MIN_CELL_AREA: 

                if preview:
                    plt.figure()
                    plt.title('Point distance')
                    plt.plot(contour[:,0,0], contour[:,0,1], '--')
                    plt.plot(ca[:,0,0], ca[:,0,1], '-o')
                    plt.plot(cb[:,0,0], cb[:,0,1], '-o')
                    plt.plot(contour[row,0,0], contour[row,0,1], 'co')
                    plt.plot(contour[col,0,0], contour[col,0,1], 'co')
                    plt.axis('equal')

                return split_contour_by_point_distance(ca, min_distance=min_distance, preview=preview) + split_contour_by_point_distance(cb, min_distance=min_distance, preview=preview)

    return [contour]





'''
Calcualtes curvature between all consecutive points in x,y arrays. 
Assumes shape is closed. 
Returns 0 when three points form straight line, negative when curving outword and positive when curving inword.
'''
def curvature(contour):

    x,y = contour[:,0,0], contour[:,0,1]
    dx = np.diff(x, prepend=x[-1])
    dy = np.diff(y, prepend=y[-1])

    angles = np.arctan2(dx, dy)
    angles = np.mod(angles, 2*np.pi)
    da = np.diff(angles, append=angles[0])
    da = np.mod(da+np.pi, 2*np.pi)-np.pi

    return da

from rdp import rdp
import itertools

def split_contour_by_curvature(contour, preview=False):

    # if contour.ndim == 3: contour = contour[:,0,:] # from opencv, thre is an empty second dimension we can get rid of

    simplified_contour = rdp(contour, epsilon=0.8)
    # simplified_contour = filter(contour, 2)

    da = curvature(simplified_contour)
    convex_corners = da < -np.pi/6 # if corner is concave with more than 30 degrees

    convex_corners_indices = convex_corners.nonzero()[0] # get indices of convex corners


    # loop over all unique combinatinos of two concave corner indixes
    best_split_contours = None # check all and pick global optimum
    best_split_separation = MIN_POINT_SEPARATION + 1 # init to some value larger than min separation limit
    for i0, i1 in itertools.combinations(convex_corners_indices, 2):
        
        if abs(i0-i1) < 5: continue # valid contours must have more points than this
        separation = np.linalg.norm(simplified_contour[i0] - simplified_contour[i1])
        if separation > MIN_POINT_SEPARATION: continue # points muse be closer than this
        if separation >= best_split_separation: continue # not interested in solution if we have found one with closer points before
        
        # split to new proposed contours
        c_a, c_b = split_at_indices(contour=simplified_contour, i0=i0, i1=i1)
        
        # check areas are sufficiently large
        if contour_to_area(c_a) < MIN_CELL_AREA: continue
        if contour_to_area(c_b) < MIN_CELL_AREA: continue

        best_split_contours = (c_a, c_b)
        best_split_separation = separation

    if best_split_contours:
        c_a, c_b = best_split_contours

        if preview:
            plt.figure()
            plt.title('Curvature')
            plt.plot(contour[:,0,0], contour[:,0,1], '--')
            plt.plot(c_a[:,0,0], c_a[:,0,1], '-o')
            plt.plot(c_b[:,0,0], c_b[:,0,1], '-o')
            plt.plot(simplified_contour[i0,0,0], simplified_contour[i0,0,1], 'co')
            plt.plot(simplified_contour[i1,0,0], simplified_contour[i1,0,1], 'co')
            plt.axis('equal')
        
        # recurse on sub-contours in case they can be split into more contours
        return split_contour_by_curvature(c_a, preview=preview) + split_contour_by_curvature(c_b, preview=preview)

    if preview:
        plt.figure()
        plt.plot(contour[:,0,0], contour[:,0,1], '--')
        plt.axis('equal')

    return [contour]



def mask_from_contour(c, padding):
    c_min = np.min(c[:,0,:],0) - padding
    c_max = np.max(c[:,0,:],0) + padding + 1 # add one to make padding symetrical on all sides
    size = c_max-c_min

    mask = np.zeros(size[::-1])
    cv.drawContours(mask, contours=[c-c_min], contourIdx=0, color=1, thickness=cv.FILLED)
    return mask, c_min # c_min is shifting of frame

# # measure max width of cell, in um
# def max_width_of_mask(mask, debug=False):
#     distance_img = ndimage.morphology.distance_transform_edt(mask)
#     return 2*np.max(distance_img) * UM_PER_PIXEL # distance transform gives distance to closest edge -> 2x for full width

# # measure length of skeletons, in um 
# def length_of_mask(mask, debug=False):
#     fil = FilFinder2D(mask, mask=mask)
#     fil.medskel(verbose=False)
#     fil.analyze_skeletons(branch_thresh=400 * u.pix, skel_thresh=2 * u.pix, prune_criteria='length', verbose=False)

#     if debug:
#         plt.figure()
#         plt.imshow(mask)
#         # plt.contour(, colors='r', linewidths=1)
#         plt.contour(fil.skeleton_longpath, colors='r', linewidths=1)
#         plt.axis('off')
#         plt.show()
#         print(fil.branch_lengths())

#     return np.sum(fil.branch_lengths()) * UM_PER_PIXEL

# # Uses skeleton to compute, but is quite slow
# def length_and_width_of_contour(c, debug=False):
#     mask, _ = mask_from_contour(c, padding=1) # smallest padding with complete zero-border
#     return length_of_mask(mask, debug), width_of_mask(mask, debug)

# def lengths_and_widths_of_contours_bounding_box(contours_ts):
#     min_area_rects_ts = twoD_stats(cv.minAreaRect, contours_ts)
#     return [[r[1][0] * UM_PER_PIXEL for r in rs] for rs in min_area_rects_ts], [[r[1][1] * UM_PER_PIXEL for r in rs] for rs in min_area_rects_ts]

# def lengths_and_widths_of_contours(contours_ts):
#     lws = twoD_stats(length_and_width_of_contour, contours_ts)
#     return [[p[0] for p in ps] for ps in lws], [[p[1] for p in ps] for ps in lws]

# def lengths_and_widths_of_contours(contours_ts):
#     a = twoD_stats(length_and_width_of_contour, contours_ts)
#     return list(zip(*[(zip(*b)) for b in a])) # convert list of list of touples to touple of list of lists



# Dilate contour by converting to mask, dilating and converting back to contour
def dilate_contour(c):
    mask, c_min = mask_from_contour(c, padding=5)
    mask = cv.dilate(mask, kernel=k5_circle, iterations=1)
    ca, _ = cv.findContours(mask.astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return ca[0]+c_min

# Compute all properties at once to avoid re-computing values
def ss_stats_from_contour(contour):
    mask, _ = mask_from_contour(contour, padding=1) # smallest padding with complete zero-border
    distance_img = ndimage.morphology.distance_transform_edt(mask)

    area = cv.contourArea(contour)
    dist_sum = np.sum(distance_img)
    max_width = 2*np.max(distance_img) # distance transform gives distance to closest edge -> 2x for full width
    
    A,E = area, dist_sum
    mean_width_simple = 4*E/A
    mean_width_roots = np.roots([np.pi/48/A, 0, -1/4, E/A])
    mean_width = abs(min(mean_width_roots, key=lambda x: abs(x-mean_width_simple))) # pick root closest to simple solution
    # print(mean_width_roots, mean_width_simple, mean_width)

    length = area/mean_width + mean_width*(1-np.pi/4)
    
    return (
        area * UM_PER_PIXEL**2,
        dist_sum * UM_PER_PIXEL**3,
        length * UM_PER_PIXEL,
        mean_width * UM_PER_PIXEL,
        max_width * UM_PER_PIXEL, 
        length/mean_width,
        length/max_width,
    )

SS_STATS_KEYS = [
    'ss_area',
    'ss_dist_sums',
    'ss_length',
    'ss_width',
    'ss_max_width',
    'ss_aspect_ratio',
    'ss_aspect_ratio_max_width',
]

# Helper to convert list of list of touple of stats to dict of list of lists, where dict keys are right for export
def ss_stats_from_contours_ts(contours_ts):
    a = twoD_stats(ss_stats_from_contour, contours_ts)
    b = list(zip(*[(zip(*b)) for b in a])) # convert list of list of touples to touple of list of lists
    return {k: v for k, v in zip(SS_STATS_KEYS, b)}


def id_from_frame_and_outline(f, contour):
    # y,x = centroid(contour)
    # y0,x0 = centroid(contour)
    pl_contour = contour[:,0].astype(float)
    x,y = polylabel_pyo3.polylabel_ext_np(pl_contour, 1.0)
    x,y = round(x), round(y)
    # print(x, x0, '-', y, y0)
    y_max, x_max = f.shape
    if 0 <= y < y_max and 0 <= x < x_max:
        return f[y,x]
    return 0
    
'''
Only keeps top-level contours above threshold size
'''
def contour_filter(cs, hierarchy, min_area):
    
    filter = np.array([contour_to_area(c) > min_area for c in cs])

    if not isinstance(hierarchy, type(None)):
        filter *= hierarchy[0][:,3] < 0 # must be index -1 (top level)
    
    return [cs[i] for i, included in enumerate(filter) if included]



### Funcitons for extracting stats
    
def oneD_stats(func, timeseries):
    return [func(a) for a in timeseries]

def twoD_stats(func, timeseries):
    #return [np.apply_along_axis(func, 0, a) for a in timeseries]
    return [[func(b) for b in a] for a in timeseries]

### Contour analysis helpers

def contour_to_area(contour):
    return cv.contourArea(contour) * UM_PER_PIXEL**2

def contour_to_arc_length(contour):
    return cv.arcLength(contour, closed=True) * UM_PER_PIXEL

def centroid(contour):
    m = cv.moments(contour)
    m00 = m["m00"] if m["m00"] != 0 else 1 # guard against division by zero
    x = int(m["m01"] / m00)
    y = int(m["m10"] / m00)
    return x, y

def cell_distance_from_colony_border(cell_centroids, colony_contours, frame_shape):
    colony_mask = np.zeros(frame_shape)
    cv.drawContours(colony_mask, contours=colony_contours, contourIdx=-1, color=1, thickness=cv.FILLED)
    colony_edt = ndimage.distance_transform_edt(colony_mask)
    # flatten from multidimentional to 1d array for each index
    return [colony_edt[point[0], point[1]] * UM_PER_PIXEL for point in cell_centroids]

def contours_in_box(contours, bound):
    '''
    return boolean array indicating if relevant conours is fully inside the boinding box.
    boxes defined as [x,y,width,height]
    '''
    
    rects = np.array([cv.boundingRect(c) for c in contours])
    
    left = rects[:,0] > bound[0]
    top = rects[:,1] > bound[1]
    right = rects[:,0] + rects[:,2] < bound[0] + bound[2]
    bottom = rects[:,1] + rects[:,3] < bound[1] + bound[3]
    
    return left * right * top * bottom

def on_border(contour, bound):
    rect = cv.boundingRect(contour)

    left = rect[0] > bound[0] # true if inside left bound
    top = rect[1] > bound[1]
    right = rect[0] + rect[2] < bound[0] + bound[2]
    bottom = rect[1] + rect[3] < bound[1] + bound[3]

    return not (left and top and right and bottom) # if all ar not inside

def contour_intersect(original_image, contour1, contour2):
    # Two separate contours trying to check intersection on
    contours = [contour1, contour2]

    # Create image filled with zeros the same size of original image
    blank = np.zeros(original_image.shape[0:2])

    # Copy each contour into its own image and fill it with '1'
    image1 = cv.drawContours(blank.copy(), contours, 0, 1)
    image2 = cv.drawContours(blank.copy(), contours, 1, 1)

    # Use the logical AND operation on the two images
    # Since the two images had bitwise and applied to them,
    # there should be a '1' or 'True' where there was intersection
    # and a '0' or 'False' where it didn't intersect
    intersection = np.logical_and(image1, image2)

    # Check if there was a '1' in the intersection
    return intersection.any()

# dimensionless number. Circle = 1. Larger number -> less uneven shape/surface.
def perimeter_area_ratio(arc_lengths, areas):
    return arc_lengths**2/areas/(4*np.pi)



### Z-stack related

def focus_score(frame):
    f = cv.GaussianBlur(frame, (5, 5), 0)
    l = cv.Laplacian(f, cv.CV_32S, ksize=5)
    l = np.absolute(l)
    return l.mean()

def focus_score_fft(frame):
    rows, cols = frame.shape
    r_max = min(rows, cols)//2
    frame = frame[:r_max, :r_max]
    frame = np.abs(np.fft.fft2(frame))
    
    # Exploit non-normalized power spectrum for 2d field. 
    # Low frequency components have low area, high frequency have large area
    # Thus, sum becomes weighted in favour of high-frequency components
    return frame.mean() # maximize power spectrum

def best_index_for_stack(stack):
    scores = [focus_score(f) for f in stack]
    return scores.index(max(scores))

def best_indices(stacks):
    return [best_index_for_stack(stack) for stack in stacks]

def best_frames_from_z_stack(stacks):
    if len(stacks[0]) == 1: # no z-stack used, for speedup
        return [s[0] for s in stacks] 

    selection = best_indices(stacks)
    return [stacks[i][idx] for i, idx in enumerate(selection)]

def best_frame_for_stack(movie, stack_indices):
    stack_frames = [movie[int(i)] for i in stack_indices]
    best_index = best_index_for_stack(stack_frames)
    print(f'Checking stack indices {stack_indices}, best is {best_index}')
    return stack_frames[best_index], stack_indices[best_index]

def index_scores(stacks):
    return [[focus_score(f) for f in stack] for stack in stacks]

# def z_stack_series(m, start, number_of_stacks=None, z_stack_size=7, consecutive_images=2, interval=2*7*60*4):
#     indices = range(start, len(m), interval)
#     if number_of_stacks:
#         indices = indices[:number_of_stacks]

#     stacks = [m[i:i+consecutive_images*z_stack_size:consecutive_images] for i in indices] # z-stacks
#     times = [m.frame_time(i, true_time=True) for i in indices]
#     return stacks, times, indices



WINDOW_SIZE = 200 # 200 is close to GCD of 3208x2200, 240 is gcd of pixel counts (1920x1200)
BLUR_SIZE = 51

class MemoizeKernelWindow:
    def __init__(self, f):
        self.f = f
        self.memo = {}
    def __call__(self, xs, ys, shape):
        key = (xs.start, xs.stop, ys.start, ys.stop, shape)
        if not key in self.memo:
            self.memo[key] = self.f(xs, ys, shape)
        return self.memo[key]

@MemoizeKernelWindow
def window_kernel(xs, ys, shape):
    k = np.zeros(shape).astype(np.uint16)
    k[ys,xs] = 255
    # k = cv.GaussianBlur(k, (51, 51), 0)
    k = cv.blur(k, (BLUR_SIZE, BLUR_SIZE))
    k = k.astype(np.float16)/255.0
    # plot_frame(k.astype(np.uint16), 'kernel', plot=xs.start == 0 and ys.start == 0)
    return k


### Colors


def color_for_number(number):
    return ColorHash(number).rgb if number != -1 else (0,0,255)
    



### Plotting


def plot_frame(f, dinfo, contours=None, new_figure=True, contour_thickness=1, contour_color_function=None, contour_labels=None):
    
    if not dinfo.live_plot and not dinfo.file_plot: # for performance
        return 
    
    if f.dtype != np.uint8:
        f = np.uint8(norm(f))

    if contours:
        f = np.stack((f,)*3, axis=-1)
        for i, _ in enumerate(contours):
            color = color_for_number(i) if contour_color_function == None else contour_color_function(i, contours[i])
            f = cv.drawContours(f, contours, contourIdx=i, color=color, thickness=contour_thickness)

        if contour_labels:
            for label, contour in zip(contour_labels, contours):
                x,y = centroid(contour)
                f = text_on_frame(f, label, (x+10,y+10), dinfo.font_file)

    if dinfo.crop != None:
        (x0,x1), (y0, y1) = dinfo.crop
        f = f[y0:y1, x0:x1]

    if dinfo.live_plot:
        if new_figure:
            plt.figure(figsize=(25, 10), dpi=80)

        plt.imshow(f, cmap='gray')
        plt.title(dinfo.label, color='w')
        plt.xticks([]), plt.yticks([])
        plt.tight_layout()

    if dinfo.file_plot:
        im = Image.fromarray(np.uint8(f))
        im.save(os.path.join(dinfo.image_dir, dinfo.label + '.tif'))
    

def frame_with_cs_ss_offset(frame, cs_contours, cs_ids, ss_contours, ss_ids, offset, cs_on_border, ss_stroke=1):
    f = norm(frame).astype(np.uint8)
    f = np.stack((f,)*3, axis=-1)

    for i, c, on_border in zip(cs_ids, cs_contours, cs_on_border):
        cv.drawContours(f, contours=[c], contourIdx=0, color=color_for_number(i), thickness=8 if on_border else 2)
    for i, c in zip(ss_ids, ss_contours):
        cv.drawContours(f, contours=[c], contourIdx=0, color=color_for_number(i), thickness=ss_stroke)
    
    if not isinstance(offset, type(None)):
        offset = np.append(offset, [0]) # add third rgb dimension
        f = scipy.ndimage.shift(f, shift=offset, cval=0)

    return f

'''
Plots entire stack at plt from matplotlib.
TODO: make file output from matplotlib figures, not just single frames as now
'''
def plot_stack(fs, dinfo):
    fig = plt.figure(figsize=(50, 10), dpi=180)
    gs = fig.add_gridspec(1, len(fs), hspace=0, wspace=0)

    scores = [focus_score(f) for f in fs]
    scores2 = [focus_score_fft(f) for f in fs]
    labels = [f'{i}: {s:.1e}, {s2:.1e}   {"b1" if max(scores)==s else ""} {"b2" if max(scores2)==s2 else ""}' for i, (s, s2) in enumerate(zip(scores, scores2))]

    for label, f, ax in zip(labels, fs, gs.subplots(sharey='row')):
        plt.sca(ax)
        plot_frame(f, dinfo.append_to_label(label), new_figure=False)



'''
Plot contour without background image
'''
def plot_contour(c):
    xs = c[:,0]
    ys = c[:,1]
    xs = np.append(xs, xs[:1])
    ys = np.append(ys, ys[:1])

    plt.gca().set_aspect('equal', 'box')
    plt.plot(xs,ys, 'o-')


### Debug visualizations

'''
Add text to frame
'''
def text_on_frame(frame, text, position, font_file):
    image = Image.fromarray(frame, 'RGB' if len(frame.shape) == 3 else None)
    canvas = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(font_file, 30)
    except OSError:
        # font = ImageFont.truetype("Lato-Regular.ttf", 30)
        font = ImageFont.load_default()
    canvas.text(position, f'{text}', (225, 225, 225), font=font)
    return np.asarray(image)


def translate_contours(contours, offset):
    return [contour+offset for contour in contours]


def masks_to_movie(frames_ts, cs_contours_ts, cs_ids_ts, ss_contours_ts, ss_ids_ts, cumulative_offset_ts, cs_on_border_ts, frame_labels, times, ss_stroke, dinfo, font_file='arial.ttf', output_frames=False):
        FPS = 5
        out = cv.VideoWriter(os.path.join(dinfo.video_dir, f'{dinfo.label}.mp4'), cv.VideoWriter_fourcc(*'mp4v'), FPS, frames_ts[0].shape[::-1])

        logging.debug(f'{len(frames_ts)}, {len(cs_contours_ts)}, {len(cs_ids_ts)}, {len(ss_contours_ts)}, {len(ss_ids_ts)}, {len(cumulative_offset_ts)}, {len(cs_on_border_ts)}, {len(times)}')

        ### Generate frames with colony and single-cell annotation
        for f, cs_contours, cs_ids, ss_contours, ss_ids, cumulative_offset, cs_on_border, frame_label, time \
            in zip(frames_ts, cs_contours_ts, cs_ids_ts, ss_contours_ts, ss_ids_ts, cumulative_offset_ts, cs_on_border_ts, frame_labels, times):
            debug_frame = frame_with_cs_ss_offset(
                frame=f, 
                cs_contours=cs_contours, 
                cs_ids=cs_ids,
                ss_contours=ss_contours, 
                ss_ids=ss_ids,
                offset=cumulative_offset,
                cs_on_border=cs_on_border,
                ss_stroke=ss_stroke,
            )

            ### Add text to frames
            debug_label = f'{dinfo.label}, {frame_label}'
            time_label = f'{time//(60*60):02.0f}:{(time//60) % 60:02.0f}:{time % 60:02.0f}'
            time_label_full = f'time = {time_label}'
            debug_frame = text_on_frame(debug_frame, debug_label, position=(10, 20), font_file=font_file)
            debug_frame = text_on_frame(debug_frame, time_label_full, position=(10, 50), font_file=font_file)

            out.write(debug_frame)

            if output_frames: 
                # output full frame
                time_label = f'full_t{time_label.replace(":", ".")}'
                plot_frame(debug_frame, dinfo.append_to_label(time_label).with_file_plot(True))

                debug_frame_no_offset = frame_with_cs_ss_offset(
                    frame=f, 
                    cs_contours=cs_contours, 
                    cs_ids=cs_ids,
                    ss_contours=ss_contours, 
                    ss_ids=ss_ids,
                    offset=[0 for f in cumulative_offset],
                    cs_on_border=cs_on_border,
                    ss_stroke=ss_stroke,
                )

                # output slice of frame that follows colony
                for contour, id, on_border in zip(cs_contours, cs_ids, cs_on_border):
                    if on_border: continue # skip colonies on border
                    
                    PADDING = 10 # px
                    x,y,w,h = cv.boundingRect(contour)
                    plot_frame(debug_frame_no_offset[y-PADDING:y+h+PADDING, x-PADDING:x+w+PADDING], dinfo.append_to_label(f'cid{id}_{frame_label}_{time_label}'))

        out.release()



COLONY_OUTPUT_SIZE = 128

def export_masks_for_first_generation(cs_contours_ts, names_ts, frames_ts, label, output_folder):
    
    if output_folder == None: return
    
    completed = set()
    for cs_contours, names, frame in zip(cs_contours_ts, names_ts, frames_ts):
        for cs, name in zip(cs_contours, names):
            if '.' in name: continue
            if name in completed: continue

            completed.add(name)
            
            # make image mask with ones and zeros
            mask = np.zeros_like(frame)
            cv.drawContours(mask, contours=[cs], contourIdx=0, color=1, thickness=cv.FILLED)
            
            # and with frame
            masked_full_frame = np.multiply(frame, mask)
            x,y,w,h = cv.boundingRect(cs)
            
            output_frame = np.zeros((COLONY_OUTPUT_SIZE, COLONY_OUTPUT_SIZE))
            
            x0, y0 = (COLONY_OUTPUT_SIZE-w)//2, (COLONY_OUTPUT_SIZE-h)//2
            if x0 < 0:
                logging.debug(f'Cannot fit mask in x for name {name} in frame, outline is {x0=}, {x=}, {w=}')
                x += (w-COLONY_OUTPUT_SIZE)//2
                x0, w = 0, COLONY_OUTPUT_SIZE
                logging.debug(f'After update, {name} outline is {x0=}, {x=}, {w=}')
            if y0 < 0:
                logging.debug(f'Cannot fit mask in x for name {name} in frame, outline is {y0=}, {y=}, {h=}')
                y += (h-COLONY_OUTPUT_SIZE)//2
                y0, h = 0, COLONY_OUTPUT_SIZE
                logging.debug(f'After update, {name} outline is {y0=}, {y=}, {h=}')
            try:
                output_frame[y0:y0+h, x0:x0+w] = masked_full_frame[y:y+h, x:x+w]

                im = Image.fromarray(np.uint8(output_frame))
                im.save(os.path.join(output_folder, f'{label}_n{name}_x{x}_y{y}.jpg'))
            
            except:
                logging.exception('Could not generate mask frame')