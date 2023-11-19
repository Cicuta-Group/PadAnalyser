import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2 as cv

def get_contours_from_labeled_frame(frame):
    max_label = np.max(frame)  # Find the maximum label value
    contours_list = []  # List to store contours for all regions

    # Iterate through each label value to find contours for each region
    for i in range(1, max_label + 1):
        # Create a binary mask for the current region
        mask = np.uint8(frame == i)
        
        # Find contours for the current region
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # Append contours to the list; assuming we want all contours
        contours_list.extend(contours)

    return contours_list


# Function to compute IoU
def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union

def compute_matched_iou(predicted_mask, ground_truth_mask, threshold=0.5):
    """
    Compute the Matched IoU for labeled frames (masks) of segmented cells.

    Parameters:
    predicted_mask (np.array): 2D array where each cell is represented by a unique integer label.
    ground_truth_mask (np.array): 2D array with the ground truth labels.
    iou_threshold (float): Threshold value to consider a predicted cell as a true positive.

    Returns:
    float: The average Matched IoU for all true positive predicted cells.
    """
    
    # Find unique labels in both masks
    pred_labels = np.unique(predicted_mask)
    gt_labels = np.unique(ground_truth_mask)
    
    # Remove background label if it exists
    pred_labels = pred_labels[pred_labels != 0]
    gt_labels = gt_labels[gt_labels != 0]
    
    ious = []
    
    # Iterate over each predicted label
    for gt_label in gt_labels:
        gt_mask = (ground_truth_mask == gt_label)
        
        intersecting_labels = np.unique(predicted_mask[gt_mask])
        intersecting_labels = intersecting_labels[intersecting_labels != 0]

        matched_iou = 0
        matched_pred_label = None

        # Iterate over each ground truth label
        for pred_label in intersecting_labels:
            pred_mask = (predicted_mask == pred_label)
            
            # Calculate intersection and union
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            iou = intersection / union if union > 0 else 0
            
            # Check if this is the highest IoU for this predicted label
            if iou > matched_iou:
                matched_iou = iou
                matched_pred_label = pred_label
        
        # If the matched IoU is above the threshold, add it to the list
        if matched_iou >= threshold:
            ious.append(matched_iou)
            
            # Remove the matched ground truth label to prevent it from matching with another predicted label
            pred_labels = pred_labels[pred_labels != matched_pred_label]
    
    # Calculate average IoU for all matches
    average_iou = np.mean(ious) if ious else 0
    
    return average_iou


def compute_dice(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    return 2. * intersection / (mask1.sum() + mask2.sum())

def compute_precision_recall(mask1, mask2):
    TP = np.logical_and(mask1, mask2).sum()
    FP = np.logical_and(mask1, np.logical_not(mask2)).sum()
    FN = np.logical_and(np.logical_not(mask1), mask2).sum()
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    return precision, recall

def compute_accuracy(mask1, mask2):
    correct_preds = (mask1 == mask2).sum()
    total = mask1.size
    return correct_preds / total

def compute_fpr_fnr(mask1, mask2):
    FP = np.logical_and(mask1, np.logical_not(mask2)).sum()
    FN = np.logical_and(np.logical_not(mask1), mask2).sum()
    TN = np.logical_and(np.logical_not(mask1), np.logical_not(mask2)).sum()
    FPR = FP / (FP + TN) if FP + TN > 0 else 0
    FNR = FN / (FN + mask2.sum()) if FN + mask2.sum() > 0 else 0
    return FPR, FNR


# input: labeled frames
def calculate_segmentation_errors(predicted, ground_truth):
    """
    Calculate the number of segmentation errors per cell. Based on metric used in Omnipose paper. 
    
    Parameters:
    - predicted: 2D array where each unique value represents a predicted cell.
    - ground_truth: 2D array where each unique value represents a ground-truth cell.
    
    Returns:
    - error_count: Total number of segmentation errors.
    - errors_per_cell: Dictionary with the number of errors for each ground-truth cell.
    """
    # Get unique labels for predicted and ground-truth cells
    predicted_cells = np.unique(predicted)
    ground_truth_cells = np.unique(ground_truth)
    
    # Remove the background label (0) if present
    predicted_cells = predicted_cells[predicted_cells != 0]
    ground_truth_cells = ground_truth_cells[ground_truth_cells != 0]
    
    # Initialize the error count and dictionary to keep track of errors per ground-truth cell
    error_count = 0
    errors_per_cell = {cell: 0 for cell in ground_truth_cells}
    
    predicted_cells_with_match = set()
    
    # Iterate over each ground-truth cell
    for gt_cell in ground_truth_cells:
        # Create a mask for the current ground-truth cell
        gt_mask = ground_truth == gt_cell
        
        # Identify predicted cells that intersect with the ground-truth cell
        intersecting_cells = np.unique(predicted[gt_mask])
        intersecting_cells = intersecting_cells[intersecting_cells != 0]

        matches = 0
        # Iterate over each predicted cell
        for pred_cell in intersecting_cells:
            # Create a mask for the current predicted cell
            pred_mask = predicted == pred_cell
            pred_area = np.sum(pred_mask)

            # Calculate the overlap area between the predicted cell and ground-truth cell
            overlap_area = np.sum(pred_mask & gt_mask)
            
            # Calculate the overlap ratio
            overlap_ratio = overlap_area / pred_area
            
            # Check if the predicted cell should be assigned to the ground-truth cell
            if  overlap_ratio >= 0.75:
                matches += 1
                predicted_cells_with_match.add(pred_cell)

        # If there are surplus matches, count them as errors
        if matches > 1:
            errors_per_cell[gt_cell] = matches - 1
            error_count += matches - 1
        # If there is no match, count it as an error
        elif matches == 0:
            errors_per_cell[gt_cell] = 1
            error_count += 1
            
    predicted_cells_without_match = len(predicted_cells) - len(predicted_cells_with_match)
    # print(f'Predicted cells without match: {predicted_cells_without_match}, error_count: {error_count}')
    return error_count + predicted_cells_without_match, errors_per_cell


# input: labeled frames
def calculate_colony_segmentation_errors(predicted_colonies, ground_truth):
    """
    Calculate the number of erronous segmented colnies.
    We add 1 to the error count if a colony does not have any cells.

    Returns:
    - error_count: Total number of segmentation errors.
    - errors_per_cell: Dictionary with the number of errors for each ground-truth colony.
    """
    # Get unique labels for predicted and ground-truth cells
    predicted_colonies = np.unique(predicted_colonies)
    ground_truth_cells = np.unique(ground_truth)
    
    # Remove the background label (0) if present
    predicted_colonies = predicted_colonies[predicted_colonies != 0]
    ground_truth_cells = ground_truth_cells[ground_truth_cells != 0]        

    # Initialize the error count and dictionary to keep track of errors per ground-truth cell
    error_count = 0
    cells_in_colonies = set()

    # Iterate over each ground-truth cell
    for pred_col in predicted_colonies:
        
        # Create a mask for the current ground-truth cell
        pred_mask = predicted_colonies == pred_col 
        
        # Identify predicted cells that intersect with the ground-truth cell
        intersecting_cells = np.unique(ground_truth[pred_mask])

        cells_in_colony = len(intersecting_cells) - 1 # -1 to remove background label
        if cells_in_colony > 0:
            error_count += 1
            cells_in_colonies.add(cells_in_colony)

    cells_without_colonies = len(ground_truth_cells) - len(cells_in_colonies)    
    errors_per_colony = error_count / len(predicted_colonies)

    return error_count, errors_per_colony, cells_without_colonies



def compute_cell_areas(labeled: list[tuple]) -> pd.DataFrame:
    return np.bincount(labeled.ravel())[1:]



def compute_metrics(f: np.ndarray, a: np.ndarray, label: str, parameters: dict, a_cell_areas: list = None) -> dict:
    
    error_count, errors_per_cell = calculate_segmentation_errors(f, a)
    iou = compute_iou(f, a)
    # matched_iou = 0 # Benchmarking.compute_matched_iou(f_labeled, a_labeled, threshold=0.5)
    
    cell_count_f = np.max(f)
    cell_count_a = np.max(a)
    f_cell_areas = compute_cell_areas(f)
    if a_cell_areas is None: a_cell_areas = compute_cell_areas(a) # can be pre-computed to save time

    return {
        'label': label,
        **parameters,
        'error_count': error_count,
        'errors_per_cell': errors_per_cell,
        'iou': iou,
        # 'matched_iou': marched_iou,
        'f_cell_count': cell_count_f,
        'a_cell_count': cell_count_a,
        'f_cell_areas': f_cell_areas,
        'a_cell_areas': a_cell_areas,
    }


def df_from_results(results):
    df = pd.DataFrame(results)

    df['species'] = df['label'].apply(lambda x: x.split('_')[0])
    df['Cell count'] = df['errors_per_cell'].apply(lambda x: len(x))
    df['Mean errors per cell'] = df['errors_per_cell'].apply(lambda x: np.mean(np.array(list(x.values()))))

    df['mean_f_cell_area'] = df['f_cell_areas'].apply(lambda x: np.mean(x))
    df['mean_a_cell_area'] = df['a_cell_areas'].apply(lambda x: np.mean(x))
    df['Area error'] = df['mean_f_cell_area'] / df['mean_a_cell_area'] - 1 # 0 is perfect match, negative is underestimation, positive is overestimation

    return df



def compare(f, a, l, plot_area_histogram=False):
    
    # Compute metrics
    iou = compute_iou(f, a)
    dice = compute_dice(f, a)
    precision, recall = compute_precision_recall(f, a)
    accuracy = compute_accuracy(f, a)
    fpr, fnr = compute_fpr_fnr(f, a)

    # Compute cell statistics
    cell_count_f = np.max(f)
    cell_count_a = np.max(a)
    f_cell_areas = compute_cell_areas(f)
    a_cell_areas = compute_cell_areas(a)
    
    mean_cell_area_f = np.mean(f_cell_areas)
    mean_cell_area_a = np.mean(a_cell_areas)
    
    if plot_area_histogram:
        plt.figure(figsize=(6,4), dpi=300)
        plt.hist(f_cell_areas, bins=100, alpha=0.5, label='Seg')
        plt.hist(a_cell_areas, bins=100, alpha=0.5, label='Annotated')
        plt.legend()
        plt.title(f'Cell area distribution {l}')
        plt.xlabel('Cell area (pixels)')
        plt.ylabel('Count')
        plt.show()

    # Create dataframe inside the function
    data = {
        "Label": [l],

        "IOU": [iou],
        "Dice": [dice],
        "Precision": [precision],
        "Recall": [recall],
        "Accuracy": [accuracy],
        "FPR": [fpr],
        "FNR": [fnr],

        "Cell Count (Seg)": [cell_count_f],
        "Cell Count (Annotated)": [cell_count_a],
        "Mean Cell Area (Seg)": [mean_cell_area_f],
        "Mean Cell Area (Annotated)": [mean_cell_area_a]
    }
    
    return pd.DataFrame(data)

def print_dict(d, end='\n'):
    for k,v in d.items():
        
        v_str = None
        if isinstance(v, list):
            pass
        elif isinstance(v, int):
            v_str = str(v)
        elif isinstance(v, float):
            v_str = f'{v:.2f}'
        
        if v_str:
            print(f'{k}: {v_str}', end=' | ')

    print('', end=end)