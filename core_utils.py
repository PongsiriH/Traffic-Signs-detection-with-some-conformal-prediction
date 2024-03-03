import numpy as np
import pandas as pd
from general import *
from core import *

"""Evaluate and results of Conformal box-wise classification
predictions_and_labels, summary_table_and_results

"""
def predictions_and_labels(model, dataset, conf_thresh, iou_nms_thresh, nonconformity_quantile, conformal_prediction_method, img_size, k_reg=None, lambda_reg=None, indices_range=None):
    predictions = []
    labels = []
    if indices_range is None:
        indices_range = range(len(dataset))
        
    for idx in indices_range: # for each image-label pair.
        data = dataset.__getitem__(idx)
        img, label, path, _ = data # label: [0, class, x, y, w, h]
        labels.append(process_label(label))
        del label
        
        # 1) forward and post-process yolo model.
        img = preprocess_image(img, (img_size, img_size))
        with torch.no_grad():
            pred = model(img)
        pred = apply_conf_thresh(pred, conf_thresh)
        pred = nms_resize_organize_etc(pred, iou_nms_thresh, dataset.img_size)
        
        # 2) use nonconformity quantile to construct a conformalized prediction set.
        pred['prediction_set_array'] = form_prediction_set(pred['class_dist'], nonconformity_quantile, conformal_prediction_method, k_reg, lambda_reg)
        pred['impath'] = path
        predictions.append(pred)
    return predictions, labels

def summary_table_and_results(labels, predictions, iou_thresh):
    """_summary_

    Args:
        labels (_type_): _description_
        predictions (_type_): _description_
        iou_thresh (_type_): to match bounding boxes for evaluation.

    Returns:
        _type_: _description_
    """
    summary_table = pd.DataFrame(columns=["target", "point prediction", "set prediction", "impath"])

    total_targets = 0
    total_correct = 0
    total_coverage = 0
    total_width = 0

    for idx, (target, prediction) in enumerate(zip(labels, predictions)):
        target_xyxy = target['xyxy']
        target['class_label'] = target['class_label'].astype(int)
        labels[idx] = target
        target_class = target['class_label']
        
        pred_xyxy = prediction['xyxy']        
        prediction_point = prediction['class_label']
        prediction_set = prediction['prediction_set_array']
        impath = prediction['impath']
        
        iou_matrix = iou(target_xyxy, pred_xyxy)
        i, j = matching_iou_hungarian(iou_matrix, iou_thresh) # maching boxes with iou
        
        assert len(pred_xyxy[j]) == len(prediction_point[j])
        assert len(prediction_point[j]) == len(prediction_set[j])
        assert len(target_xyxy[i]) == len(target_class[i])
        assert len(pred_xyxy[j]) == len(target_class[i])
        
        matched_gts = target_class[i]
        matched_point_pred = prediction_point[j]
        matched_set_pred = prediction_set[j]

        if len(pred_xyxy) == 0:
            continue

        r = convert_format(matched_gts, matched_point_pred, matched_set_pred, impath)
        summary_table = pd.concat([summary_table, r], axis=0)
        
        total_coverage += matched_set_pred[np.arange(len(matched_gts)), matched_gts.astype(int)].sum()
        total_width += prediction_set.sum().item()
        total_targets += len(matched_gts)
        total_correct += (matched_gts == matched_point_pred).sum()

    result = {
        'accuracy': total_correct / total_targets, 
        'coverage': total_coverage / total_targets, 
        'avg_width': total_width / total_targets
        }

    summary_table = summary_table.reset_index(drop=True)
    return summary_table, result

"""Conformal Risk Control utils


"""
def create_mask(boxes, width, height):
    """
    To Generate Binary mask and then used to calculate intersection region and union region.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for box in boxes:
        scaled_box = [
            int(box[0] * width), int(box[1] * height),
            int(box[2] * width), int(box[3] * height)
        ]
        mask[scaled_box[1]:scaled_box[3], scaled_box[0]:scaled_box[2]] = 1
    return mask

def intersection_over_label_area(pred_mask, label_mask):
    intersection = np.logical_and(pred_mask, label_mask)
    intersection_area = np.sum(intersection)
    label_area = np.sum(label_mask)
    return intersection_area / label_area if label_area else 0