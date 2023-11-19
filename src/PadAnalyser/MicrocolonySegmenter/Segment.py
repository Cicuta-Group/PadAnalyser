import logging
from . import ColonySegment, CellSegment, ZStack, MKSegmentUtils, DInfo
from typing import Optional
import numpy as np
import cv2 as cv

pseu = "Pseudomonas aeruginosa"
staph = "Staphylococcus aureus"
ecoli = "Escherichia coli"

species_map = {
    'pseu': pseu,
    'staph': staph,
    'ecoli': ecoli,
}

analysis_parameters = {
    staph: {
        'sigma': 2.5,
        'threshold': -2000,
        'split_factor': 0.75,
        'min_mask_size_filter': 4,
    },
    
    # ecoli: { # used in ecoli_1
    #     'sigma': 2,
    #     'threshold': -1000,
    #     'split_factor': 0.1,
    #     'min_mask_size_filter': 40,
    # },
    ecoli: { # used in ecoli_2
        'sigma': 1.5,
        'threshold': -1000,
        'split_factor': 0.3,
        'min_mask_size_filter': 40,
    },
    
    # pseu: { # used in pseu_1
    #     'sigma': 1,
    #     'threshold': -4000,
    #     'split_factor': 0.5,
    #     'min_mask_size_filter': 20,
    # },
    pseu: { # used in pseu_2
        'sigma': 1,
        'threshold': -3000,
        'split_factor': 0.5,
        'min_mask_size_filter': 20,
    },
}

def get_params_for_species(species: Optional[str]) -> dict:
    if species is None:
        species = ecoli
    if species not in analysis_parameters:
        raise ValueError(f'Unknown species: {species}')

    return analysis_parameters[species]


def segment_frame(f: np.ndarray, d: DInfo.DInfo, params: Optional[dict] = None, species: Optional[str] = None):
    
    if params is None: params = get_params_for_species(species)
    logging.info(f'Segmenting frame {d.label} as species {species} with params {params}')

    frame = ZStack.clip(f)
    frame = cv.GaussianBlur(frame, (3, 3), 0)
    frame = MKSegmentUtils.norm(frame)

    c_contours = ColonySegment.bf_via_edges(
        frame, 
        dinfo=d.append_to_label('col')
    )

    s_contours = CellSegment.bf_laplacian(
        frame,
        colony_contours=c_contours,
        dinfo=d.append_to_label('cell'),
        sigma=params['sigma'],
        ksize=7,
        threshold=params['threshold'],
        split_factor=params['split_factor'],
        min_mask_size_filter=params['min_mask_size_filter'],
    )

    # MKSegmentUtils.plot_frame(frame, dinfo=d.append_to_label('res_0_colony'), contours=c_contours, contour_thickness=2)
    # MKSegmentUtils.plot_frame(frame, dinfo=d.append_to_label('res_1_cells'), contours=s_contours, contour_thickness=cv.FILLED)
    # MKSegmentUtils.plot_frame_color_area(frame, dinfo=d.append_to_label('res_2_cell_area'), contours=s_contours)
    # MKSegmentUtils.plot_frame_color_edist(frame, dinfo=d.append_to_label('res_3_cell_edist'), cell_contours=s_contours, colony_contours=c_contours)

    return frame, c_contours, s_contours