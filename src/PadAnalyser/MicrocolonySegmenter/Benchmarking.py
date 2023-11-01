import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Function to compute IoU
def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union


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


def compute_cell_areas(labeled):
    return np.bincount(labeled.ravel())[1:]

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
