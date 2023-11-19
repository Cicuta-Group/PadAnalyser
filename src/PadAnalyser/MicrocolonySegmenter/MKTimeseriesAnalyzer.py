'''
Frame allignment and lineage tracking
'''

import numpy as np
import collections
from sklearn.neighbors import NearestNeighbors
import cv2 as cv
import logging
import os, re
from PIL import Image

from PadAnalyser.FrameSet import FrameSet
from . import MKSegmentUtils, MKLineageTracker, DInfo, ZStack, ColonySegment, CellSegment, Segment


### Frame alignment

def point_show(points, f0, dot_size=4):
    a = np.zeros_like(f0)
    for x,y in points:
        a[x-dot_size:x+dot_size,y-dot_size:y+dot_size] = 1
    
    MKSegmentUtils.plot_frame(a, title='test')

def point_show_colors(points, f0, dot_size=4):
    b = np.zeros((*f0.shape, 3))
    for i, ps in enumerate(points):
        for x,y in ps:
            b[x-dot_size:x+dot_size,y-dot_size:y+dot_size,i] = 1
    MKSegmentUtils.plot_frame(b, title='test')

def nearest_neighbour_score(f0, f1, max_x=None, max_y=None, edge_factor=0.2):
    # f1 and f2 are sets of x,y coordinates
    # Returns a normalized score of how close each point is to their nearest neighbour
    # 0 for f1 == f2
    # asymptotically decreases to zero for any separation betweeen points
    if max_x and max_y:
        x_lim_min = int(max_x*(edge_factor))
        x_lim_max = int(max_x*(1-edge_factor))
        y_lim_min = int(max_y*(edge_factor))
        y_lim_max = int(max_y*(1-edge_factor))
        
        x_points = np.logical_and(f0[:,0]>x_lim_min, f0[:,0]<x_lim_max)
        y_points = np.logical_and(f0[:,1]>y_lim_min, f0[:,1]<y_lim_max)
        f0 = f0[np.where(np.logical_and(x_points, y_points))]

    # point_show_colors(points=[f0, f01, f1], f0=frames[0]) 
    # return
    if len(f0) == 0:
        return 0.

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='euclidean').fit(f1)
    distances, indices = nbrs.kneighbors(f0) # compute distance from each filtered point to all points in f1
    score = 1/(0.01*distances**2+1)
    return np.mean(score)


def frame_offset(f0, f1, dxy, n_search, max_x, max_y, debug=False):
    # computes offset requiered for second set of points to allign with the first as best as possible

    if len(f0) == 0 or len(f1) == 0:
        return 0, 0

    bound = int(dxy*n_search)

    best_offset = 1,0
    best_score = 0 
    
    if debug:
        FILENAME = 'debug_data.txt'
        with open(FILENAME, 'w') as f:
            f.write(f'x,y,z')

    for x in range(-bound, bound, dxy):
        for y in range(-bound, bound, dxy):
            score = nearest_neighbour_score(f0, np.add(f1, [x,y]), max_x=max_x, max_y=max_y, edge_factor=0.15) 
            score -= 0.0001*np.sqrt(x**2+y**2) # bias towards smaller shift
            if debug:
                #print(f'{x=:4}  {y=:4}    {score=:.2f}')
                with open(FILENAME, 'a') as f:
                    f.write(f'{x:4}, {y:4}, {score:.4f}\n')
            if score > best_score:
                best_score = score
                best_offset = (x,y)

    #print(f'offset score: {best_score:.4f}')

    return best_offset


def offset_for_frame_pairs(ps_now, ps_next, frame_shape):
    if len(ps_now) == 0 or len(ps_next) == 0:
        return 0, 0

    # dxy=10, n_search=20 -> 1,20 works well for large offsets. Normal, assume less offset possible!

    rough_offset_x, rough_offset_y = frame_offset(ps_now, ps_next, dxy=5, n_search=20, max_x=frame_shape[0], max_y=frame_shape[1], debug=False)
    
    ps_next_updated = np.add(ps_next, [rough_offset_x, rough_offset_y])
    fine_offset_x, fine_offset_y = frame_offset(ps_now, ps_next_updated, dxy=1, n_search=10, max_x=frame_shape[0], max_y=frame_shape[1], debug=False)
   
    offset_x = rough_offset_x + fine_offset_x
    offset_y = rough_offset_y + fine_offset_y
    
    return offset_x, offset_y



def extract_colony_timeseries_data(stats, key):

    colony_stats = collections.defaultdict(list)

    for timestep in stats:

        colony_properties = timestep[key]
        colony_ids = timestep['colony_ids']

        for colony_property, colony_id in zip(colony_properties, colony_ids):
            if colony_id != -1:
                colony_stats[int(colony_id)].append(colony_property)

    return colony_stats



### NEW


def lineages_from_aligned_centroids(aligned_centroids_ts):
    
    # Strategy: 
    #  - all cells in first frame get name (use their 1-index)
    #  - search for nearest neighbour in next frame. If close enough, it gets the same index as their parent.
    #    - what is close can be detemined by distance to nearest neighbour in same timeframe..?? 
    #  - repeat. If a parent is not found, lineage number is -1

    colony_ids = np.arange(0, len(aligned_centroids_ts[0]))
    colony_ids_ts = [colony_ids]

    for last, this in zip(aligned_centroids_ts, aligned_centroids_ts[1:]):
        
        if len(last) == 0 or len(this) == 0 or len(colony_ids) == 0:
            colony_ids = []

        else:
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='euclidean').fit(last)
            distances, indices = nbrs.kneighbors(this) # compute distance from each filtered point to all points in f1
            indices = indices[:,0] # pick indices from closest neigbours

            match_indices = distances[:,0] < 10 # within x values
            colony_ids = np.take(colony_ids, indices, axis=0)
            colony_ids = np.where(match_indices, colony_ids, -1) # set non-matches to -1
        

        colony_ids_ts.append(colony_ids)

    return colony_ids_ts


def ss_ids_from_col_ids(ss_contours, cs_contours, cs_ids, f_shape, dinfo):
    ### Make ID mask from contours (each contour is filled with different integer color) -> cm
    cm = np.zeros(f_shape).astype(np.uint16)
    for i, id in enumerate(cs_ids):
        cv.drawContours(cm, contours=cs_contours, contourIdx=i, color=int(id+1), thickness=cv.FILLED)
    MKSegmentUtils.plot_frame(MKSegmentUtils.norm(cm), dinfo=dinfo.append_to_label('1_cm'))
    
    ### Compute id of single cells based on colony matrix
    ss_ids = [MKSegmentUtils.id_from_frame_and_outline(cm, c)-1 for c in ss_contours]

    return ss_ids




def analyze_time_seriess(frame_set: FrameSet, species: str, mask_folder: str, label: str, dinfo: DInfo.DInfo):

    if len(frame_set) <= 2: 
        raise Exception(f'Not enough frames to analyze for frame series {dinfo.label}, {len(frame_set)} frames')

    dinfos = [dinfo.replace_label(str(i)) for i in frame_set.get_frame_labels()]

    raw_frames_ts, times = zip(*frame_set[:])
    frames_ts, cs_contours_ts, ss_contours_ts = zip(*[Segment.segment_frame(f=f, d=d, species=species) for f, d in zip(raw_frames_ts, dinfos)])
    frame_shape = frames_ts[0].shape

    # plt.figure(figsize=(6,4), dpi=300)
    # plt.hist(frame.flatten(), bins=256, range=(0,256), log=True, histtype='stepfilled')
    # plt.title(l)
    
    # Compute contours (masks with each colony filled with different integer color)
    # cs_contours_ts = [MKSegment.bf_colony_segment(l, d.append_to_label('cs')) for l, d in zip(frames_ts, dinfos)]
    # ss_contours_ts = [MKSegment.bf_single_cell_segment(f, cs, d.append_to_label('ss')) for f, cs, d in zip(frames_ts, cs_contours_ts, dinfos)]

    # Compute centroids
    cs_centroids_ts = MKSegmentUtils.twoD_stats(MKSegmentUtils.centroid, cs_contours_ts)
    cs_centroids_ts = MKSegmentUtils.oneD_stats(np.array, cs_centroids_ts)
    
    EDGE_DIST = 15
    y_max, x_max = frame_shape
    bounding_rect = (EDGE_DIST, EDGE_DIST, x_max-2*EDGE_DIST, y_max-2*EDGE_DIST)
    cs_on_border_ts = MKSegmentUtils.twoD_stats(lambda c: MKSegmentUtils.on_border(c, bound=bounding_rect), cs_contours_ts) # if touching the edge of the frame

    ### Compute cumulative offsets for all frames and compute aligned centroids 
    frame_by_frame_offsets = [offset_for_frame_pairs(c0, c1, frame_shape) for c0, c1 in zip(cs_centroids_ts, cs_centroids_ts[1:])]
    
    cumulative_offset_ts = np.cumsum(frame_by_frame_offsets, axis=0)
    cumulative_offset_ts = np.insert(cumulative_offset_ts, 0, [0,0], axis=0) # add zero offset for zeroth index
        
    ### Determine colony lineages

    # aligned_centroids_ts = [
    #     np.add(cs_centroids, offset) if len(cs_centroids) else []
    #     for cs_centroids, offset in zip(cs_centroids_ts, cumulative_offset_ts)
    # ]
    # colony_ids_ts = lineages_from_aligned_centroids(aligned_centroids_ts) # old approach
    
    aligned_contours_ts = [
        [np.add(contour, offset[::-1]) if len(contour) else [] for contour in contours]
        for contours, offset in zip(cs_contours_ts, cumulative_offset_ts)
    ]

    id_objects = MKLineageTracker.timeseries_ids(
        cs_contours_ts=aligned_contours_ts,
        frame_shape=frame_shape,
        dinfo=dinfo,
    )
    cs_ids_ts = id_objects['ids']

    frame_labels = frame_set.get_frame_labels()

    ### Link bacteria to colony
    ss_ids_ts = [
        ss_ids_from_col_ids(ss_contours=a, cs_contours=b, cs_ids=c, f_shape=frame_shape, dinfo=dinfo.append_to_label(l)) 
        for a,b,c,l in zip(ss_contours_ts, cs_contours_ts, cs_ids_ts, frame_labels)
    ]

    logging.debug(f'{len(ss_contours_ts)}, {len(cs_contours_ts)}, {len(cs_ids_ts)}, {len(frames_ts)}')

    if dinfo.video:
        # Colony with image as background
        # MKSegmentUtils.masks_to_movie(
        #     frames_ts=frames_ts, 
        #     cs_contours_ts=cs_contours_ts, 
        #     cs_ids_ts=cs_ids_ts, 
        #     ss_contours_ts=ss_contours_ts, 
        #     ss_ids_ts=ss_ids_ts, 
        #     cumulative_offset_ts=cumulative_offset_ts, 
        #     cs_on_border_ts=cs_on_border_ts, 
        #     frame_labels=frame_labels, 
        #     times=times, 
        #     ss_stroke=1, 
        #     dinfo=dinfo.append_to_label('csi'),
        # )

        ss_unique_ids_ts = [list(range(len(ids))) for ids in ss_ids_ts]
        length = len(frames_ts)

        MKSegmentUtils.masks_to_movie(
            frames_ts=frames_ts, 
            cs_contours_ts=cs_contours_ts, 
            cs_ids_ts=cs_ids_ts,
            ss_contours_ts=ss_contours_ts, 
            ss_ids_ts=ss_unique_ids_ts, 
            cumulative_offset_ts=cumulative_offset_ts, 
            cs_on_border_ts=cs_on_border_ts, 
            frame_labels=frame_labels, 
            times=times, 
            ss_stroke=1,
            output_frames=True,
            dinfo=dinfo.append_to_label('css'),
        )


        # Colony with laplace background
        # MKSegmentUtils.masks_to_movie(
        #     frames_ts=laplacian_frames_ts, 
        #     cs_contours_ts=cs_contours_ts, 
        #     cs_ids_ts=cs_ids_ts, 
        #     ss_contours_ts=ss_contours_ts, 
        #     ss_ids_ts=ss_ids_ts, 
        #     cumulative_offset_ts=cumulative_offset_ts, 
        #     cs_on_border_ts=cs_on_border_ts, 
        #     frame_labels=frame_labels, 
        #     times=times, 
        #     ss_stroke=1, 
        #     dinfo=dinfo.append_to_label('csl'),
        # )
        
        # Single cell with image as background
        MKSegmentUtils.masks_to_movie(
            frames_ts=frames_ts, 
            cs_contours_ts=[[]]*length, 
            cs_ids_ts=[[]]*length, 
            ss_contours_ts=ss_contours_ts, 
            ss_ids_ts=ss_unique_ids_ts, 
            cumulative_offset_ts=cumulative_offset_ts, 
            cs_on_border_ts=cs_on_border_ts, 
            frame_labels=frame_labels, 
            times=times,
            ss_stroke=cv.FILLED, 
            dinfo=dinfo.append_to_label('ssi'),
        )

        # # Image background only
        # MKSegmentUtils.masks_to_movie(
        #     frames_ts=frames_ts, 
        #     cs_contours_ts=[[]]*length, 
        #     cs_ids_ts=[[]]*length, 
        #     ss_contours_ts=[[]]*length, 
        #     ss_ids_ts=[[]]*length, 
        #     cumulative_offset_ts=cumulative_offset_ts, 
        #     cs_on_border_ts=cs_on_border_ts, 
        #     frame_labels=frame_labels, 
        #     times=times, 
        #     ss_stroke=cv.FILLED, 
        #     dinfo=dinfo.append_to_label('img'),
        # ) 

        # # Laplacian images only
        # MKSegmentUtils.masks_to_movie(
        #     frames_ts=laplacian_frames_ts, 
        #     cs_contours_ts=[[]]*length, 
        #     cs_ids_ts=[[]]*length, 
        #     ss_contours_ts=[[]]*length, 
        #     ss_ids_ts=[[]]*length, 
        #     cumulative_offset_ts=cumulative_offset_ts, 
        #     cs_on_border_ts=cs_on_border_ts, 
        #     frame_labels=frame_labels, 
        #     times=times, 
        #     ss_stroke=cv.FILLED, 
        #     dinfo=dinfo.append_to_label('lap'),
        # )


    # MKSegmentUtils.export_masks_for_first_generation(
    #     cs_contours_ts=cs_contours_ts,
    #     names_ts=id_objects['names'], 
    #     frames_ts=frames_ts,
    #     label=label,
    #     output_folder=mask_folder,        
    # )

    ### Collect statistics
    # All 2D statistics is in lists of the form (time, colony/cell)

    ss_centroids_ts = MKSegmentUtils.twoD_stats(MKSegmentUtils.centroid, ss_contours_ts)
    ss_distance_from_colony_edge_ts = [
        MKSegmentUtils.cell_distance_from_colony_border(ss_centroids, colony_contours, frame_shape)
        for ss_centroids , colony_contours in zip(ss_centroids_ts, cs_contours_ts)
    ]

    return {
        'time': times,

        # Colony statistics for each colony for each timestep
        'colony_count':       MKSegmentUtils.oneD_stats(len, cs_contours_ts),
        'colony_area':        MKSegmentUtils.twoD_stats(MKSegmentUtils.contour_to_area, cs_contours_ts), # µm2
        'colony_arc_length':  MKSegmentUtils.twoD_stats(MKSegmentUtils.contour_to_arc_length, cs_contours_ts), # µm
        'colony_centroid':    cs_centroids_ts, # px,px
        'colony_on_border':   cs_on_border_ts,
        'colony_id':          cs_ids_ts, 
        'colony_ID':          id_objects['IDs'],
        'colony_name':        id_objects['names'],
        'colony_id_ID_map':   id_objects['id_ID_map'],
        'colony_ID_name_map': id_objects['ID_name_map'],

        # Single cell statistics for each cell for each timestep
                                      **MKSegmentUtils.ss_stats_from_contours_ts(ss_contours_ts),
        'ss_count':                     MKSegmentUtils.oneD_stats(len, ss_contours_ts), # overall count in frame for each timestep
        'ss_id':                        ss_ids_ts,
        'ss_centroid':                  ss_centroids_ts, # px,px
        'ss_distance_from_colony_edge': ss_distance_from_colony_edge_ts,

        # Raw information to reproduce masks
        'cs_contours':                  cs_contours_ts,
        'ss_contours':                  ss_contours_ts,
        'cumulative_offset':            cumulative_offset_ts,
        'times':                        times,
    }
