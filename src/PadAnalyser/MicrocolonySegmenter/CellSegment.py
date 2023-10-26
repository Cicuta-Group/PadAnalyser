from . import MKSegmentUtils, DInfo, CellSegmentMods

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

    if dinfo.printing:
        logging.debug(f'Raw frame info: type={f.dtype}, min={np.min(f)}, max={np.max(f)}')

    ### Include intensity from BF to get rid of cell outlines -> m1
    m1 = f
    m1 = cv.GaussianBlur(m1, (3, 3), 0)
    m1 = cv.adaptiveThreshold(m1, 1, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 81, -4) # -4, relativley low, to make better at segmenting big cells such as Mecilinam-exposed ecoli
    MKSegmentUtils.plot_frame(m1*255, dinfo=dinfo.append_to_label('3_m1'))

    # remove background based on colony locations
    colony_mask = np.zeros_like(f, dtype=np.uint8)
    cv.drawContours(colony_mask, colony_contours, -1, 1, cv.FILLED)
    m1 = m1 * colony_mask

    # remove background lonely pixels 
    kernel = np.ones((5, 5))
    isolated_pixels = convolve(m1, kernel) < 10
    m1 = m1 & ~isolated_pixels

    ## Detect contours -> contours
    m1 = cv.adaptiveThreshold(m1, 1, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 81, -4)
    MKSegmentUtils.plot_frame(m1*255, dinfo=dinfo.append_to_label('3_m11'))
    
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

    return contours