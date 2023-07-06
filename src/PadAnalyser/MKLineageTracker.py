import cv2 as cv
from collections import defaultdict
import numpy as np

from MKImageAnalysis.MKSegmentUtils import norm, id_from_frame_and_outline, plot_frame



def parent_ID(ID):
    last_index = ID.rfind('.')
    if last_index == -1: return ''
    return ID[:last_index]

# set origin_id to None if first in lineage
def new_id(origin_id, id_ID_map):
    '''
    Examples of exising: 0, 1, 2, 3, 1.0, 1.1, 2.0, 2.1, 1.0.1
    new_id request for:
        - '': -> '4'
        - '1': -> '1.2'
        - '1.0.1': -> '1.0.1.0'
    '''
    # find
    origin_ID = id_ID_map[origin_id] if origin_id != None else ''
    ID_matches = [ID for ID in id_ID_map if parent_ID(ID) == origin_ID]

    new_ID = f'{origin_ID}.{len(ID_matches)}' if origin_ID else f'{len(ID_matches)}'
    new_id = len(id_ID_map)
    id_ID_map.append(new_ID) # lands at index new_id
    return new_id

def list_duplicates(seq):
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items() if len(locs)>1)

def ids_from_col_ids(new_contours: list, cs_contours: list, cs_ids: list, id_ID_map, mask_frame, dinfo):
    ### Make ID mask from contours (each contour is filled with different integer color) -> cm
    
    # Draw on same frame buffer - that way, tracking persists even if segmentation fails for a time-step
    # mask_frame = np.zeros_like(mask_frame).astype(np.uint16) # uncomment this line to remove memory in tracking
    # IDs start at zero, so add one in image representation to keep 0 as background
    for i, id in enumerate(cs_ids):
        cv.drawContours(mask_frame, contours=cs_contours, contourIdx=i, color=int(id+1), thickness=cv.FILLED)

    plot_frame(norm(mask_frame), dinfo=dinfo.append_to_label('1_mask_frame')) #.with_file_plot()
    
    ### Compute id of single cells based on colony matrix
    new_ids = [id_from_frame_and_outline(mask_frame, c)-1 for c in new_contours]

    # give new IDs to colonies with none
    new_ids = [new_id(None, id_ID_map) if i==-1 else i for i in new_ids] 
    
    # for identical ids, assume merger has happened, give each child sub-id
    for id, indices in sorted(list_duplicates(new_ids)):
        for i in indices:
            new_ids[i] = new_id(id, id_ID_map) # give new id and keep track of relationship to parent
        
    # display results
    # plot_frame_cs_ss(
    #     f=f, 
    #     dinfo=dinfo.disabled(),
    #     cs_contours=cs_contours, 
    #     cs_ids=cs_ids, 
    #     ss_contours=ss_contours, 
    #     ss_ids=ss_ids
    # )
    return new_ids


JOIN_CHAR = '-'

def forward_ID(child_ID, ID_history, ID_name_map):
    '''
    Examples of exising: 0, 1, 2, 3, 1.0, 1.1, 2.0, 2.1, 1.0.1 -> in this list, parent have long IDs
    Maps to:                          5   1.0   3    2   0
    Request for:
        - '1.0.1' -> '0'
        - '2.1' -> '1'
        - '2.0' -> '3'
        - '1.1' -> '4'
        - '1.0' -> '0.0'
    '''
    # find 

    # parent = merged colonies
    # children = two nodes that merged into parent

    def child_of(parent, child): 
        return parent[:len(child)] == child # check if first characters match

    children = [parent_ID for parent_ID in ID_history if child_of(parent_ID, child_ID)]

    # no children found for parent node, thus give new name
    if len(children) == 0: 
        new_name = str(len([v for v in ID_name_map.values() if JOIN_CHAR not in v]))
        children.append(new_name)

    child_name = JOIN_CHAR.join(children)

    return child_name
    

# compute IDs for colonies on basis of 
def timeseries_ids(cs_contours_ts: list, frame_shape, dinfo):

    id_ID_map = [] # index id mapping to string ID

    # ID is string, ints with '.' to separate generations
    last_ids = [] # list(range(len(cs_contours_ts[-1]))) # give IDs to colonies in last frame
    id_history = [] # [last_ids] # add this to history, gives history with as many elements as there are frames
    mask_frame = np.zeros(frame_shape).astype(np.uint16)
    for t_index, (cs0, cs1) in list(enumerate(zip(cs_contours_ts, cs_contours_ts[1:])))[::-1]: # get two consecutive contour sets, start at end of video
        last_ids = ids_from_col_ids(
            new_contours=cs0,
            cs_contours=cs1,
            cs_ids=last_ids,
            id_ID_map=id_ID_map,
            mask_frame=mask_frame,
            dinfo=dinfo.append_to_label(f't{t_index}')
        )
        id_history.insert(0, last_ids)

    ID_history = [[id_ID_map[id] for id in ids] for ids in id_history]

    ID_name_map = {}

    # each input ID will map to a distinct output ID
    # start at the back will ensure we meet children before parents
    for ID in id_ID_map[::-1]:
        if ID not in ID_name_map.keys(): # need to add
            ID_name_map[ID] = forward_ID(
                child_ID=ID, 
                ID_history=ID_history, 
                ID_name_map=ID_name_map
            )

    name_history = [[ID_name_map[ID] for ID in IDs] for IDs in ID_history]

    return {
        'ids': id_history,
        'IDs': ID_history,
        'names': name_history,
        'id_ID_map': id_ID_map,
        'ID_name_map': ID_name_map,
    }


def test():
    id_ID_map = [] # index id mapping to string ID
    print(new_id(origin_id=None, id_ID_map=id_ID_map))
    print(new_id(origin_id=None, id_ID_map=id_ID_map))
    print(new_id(origin_id=None, id_ID_map=id_ID_map))
    print(new_id(origin_id=0, id_ID_map=id_ID_map))
    print(new_id(origin_id=0, id_ID_map=id_ID_map))
    print(new_id(origin_id=1, id_ID_map=id_ID_map))
    print(new_id(origin_id=0, id_ID_map=id_ID_map))
    print(new_id(origin_id=4, id_ID_map=id_ID_map))
    print(new_id(origin_id=5, id_ID_map=id_ID_map))
    print(new_id(origin_id=6, id_ID_map=id_ID_map))

    print(id_ID_map)

if __name__ == '__main__':
    test()