import os, copy, subprocess
from scipy.optimize import linear_sum_assignment
import numpy as np
import pandas as pd
import torch
import torchvision
import holoviews as hv
try:
    from yolov5.utils.general import xywh2xyxy
except:
    print("Warning for detection: Please install yolov5.")
    
def matching_iou_hungarian(iou_matrix, iou_threshold=0):
    iou_matrix = np.where(iou_matrix < iou_threshold, 0, iou_matrix)
    cost_matrix = 1 - iou_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind

def one_hot_encode(label, num_classes, device='cpu'):
    return (
        torch.nn.functional.one_hot(torch.asarray(label, dtype=torch.int64), num_classes=num_classes)
        .to(device)
        .numpy()
    )
    
def iou(box1, box2, eps=1e-7):
    """
    return ious
    numpy version of: from yolov5.utils.metrics import box_iou
    """
    # Expand dimensions to support broadcasting for intersection calculation
    box1 = np.expand_dims(box1, axis=1)  # Shape: [N, 1, 4]
    box2 = np.expand_dims(box2, axis=0)  # Shape: [1, M, 4]

    # Calculate intersection
    inter = np.maximum(0.0, np.minimum(box1[..., 2:], box2[..., 2:]) - np.maximum(box1[..., :2], box2[..., :2]))
    inter_area = inter[..., 0] * inter[..., 1]

    # Calculate areas
    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])

    # Calculate IoU
    union_area = area1 + area2 - inter_area
    iou = inter_area / (union_area + eps)

    return iou

def process_label(labels: torch.Tensor):
    labels = labels.to('cpu').numpy().copy()
    xywh = labels[..., 2:]
    xyxy = xywh2xyxy(labels[..., 2:])
    cls = labels[..., 1] # assume no label smoothing
    return {
        "xywh": xywh, 
        "xyxy": xyxy, 
        "class_label": cls
        }

def preprocess_image(img, size):
    img = torch.asarray(img).to("cuda").float() / 255.0
    img = torchvision.transforms.Resize(size)(img)
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    return img


"""
Post processing of YOLO detection's raw output.
"""
def apply_conf_thresh(pred, conf_thresh):
    """
    pred all values should be in torch.Tensor, cuda. 
    filter with low confidence score.
    """
    pred['conf_score'], _ = pred['conf_score'].max(axis=1)
    high_conf_mask = pred['conf_score'] > conf_thresh
    for key in ['xywh', 'class_dist', 'conf_score']:
        pred[key] = pred[key][high_conf_mask]
    return pred

def nms_resize_organize_etc(pred, iou_nms_thresh, img_size):
    """pred all values should be in torch.Tensor, cuda. 
    1) resize
    2) convert format from 'xywh' to 'xyxy'
    3) apply non-max suppression
    4) apply softmax to class distribution
    5) apply argmax to get the most likely class label (point-prediction).
    """
    pred['xywh'] /= img_size
    pred['xyxy'] = xywh2xyxy(pred['xywh'])
    xyxy_pred = xywh2xyxy(pred['xywh'])
    
    keep_indices = torchvision.ops.nms(xyxy_pred, pred['conf_score'], iou_nms_thresh)
    pred['xywh'] = pred['xywh'][keep_indices].to('cpu').numpy()
    pred['xyxy'] = pred['xyxy'][keep_indices].to('cpu').numpy()
    pred['conf_score'] = pred['conf_score'][keep_indices].to('cpu').numpy()
    pred['class_dist'] = pred['class_dist'][keep_indices]
    
    # Softmax
    pred['class_dist'] = torch.nn.functional.softmax(pred['class_dist'], dim=1).to('cpu').numpy()
    pred['class_label'] = pred['class_dist'].argmax(axis=1)
    return pred


"""
results into pandas.
"""
def convert_format(matched_gts, matched_point_pred, matched_set_pred, impath=None):
    result = {
        "target": matched_gts,
        "point prediction": matched_point_pred,
        "set prediction": [],
        "impath" : []
    }
    for set_pred in matched_set_pred:
        result["set prediction"].append(np.where(set_pred))
        result["impath"].append(impath)
    return pd.DataFrame(result)

def combine_tables_pandas(cal_table, new_table):
    # Concatenate 'target' and 'point prediction' directly
    combined_table = pd.concat([cal_table, new_table], axis=1)
    return combined_table


"""
Just my preference to have sklearn-like interface.
"""
class YOLO2SKLEARN:
    """sklearn-like interface"""
    def __init__(self, model):
        self.model = model
    
    def predict(self, x):
        if len(x.shape) == 3:
            x = x[None, ...] 
        
        y = self.model(x)[0]
        
        if len(x) == 1: # if only one image is passed in
            y = y[0]
            
        return {
            'xywh': y[..., :4],
            'conf_score': y[..., 4:5],
            'class_dist': y[..., 5:]
        }        
    
    def predict_proba(self, x):
        return self.predict(x)
    
    def predict_nms(self, x):
        if len(x.shape) == 3:
            x = x[None, ...] 
        
        y = self.model(x)[0]
        return non_max_suppression(y)
    
    def __call__(self, x):
        return self.predict(x)
    
    def fit(self, X):
        pass


"""
This section just for plottings.
"""
def adjust_bbox_for_flipud(bbox, img_height):
    bbox_ = copy.deepcopy(bbox)
    bbox_[:, 1] = img_height - bbox_[:, 1]
    bbox_[:, 3] = img_height - bbox_[:, 3]
    bbox_[:, [1, 3]] = bbox_[:, [3, 1]]
    return bbox_

def label_bbox(rectangles, labels, color, classes_mapping=None):
    """Create labelled rectangles for bounding boxes."""
    labelled_rects = hv.Overlay()
    for rect, label in zip(rectangles.data, labels):
        if classes_mapping:
            if isinstance(label, int) or isinstance(label, np.int32):
                label = classes_mapping[label]
            elif isinstance(label, list) or isinstance(label, np.ndarray):
                label = [classes_mapping[lbl] for lbl in label]
            else:
                raise NotImplementedError(f"Not implemented.. got {type(label)}.. {label}")
            
        # Extract the top-left corner of the rectangle for placing the label
        x, y = rect[0], rect[3]
        # Create a text label
        # text = hv.Labels([(x, y, str(label))]).opts(
        #     text_font_size='10pt', text_color=color, bgcolor='green'
        # )
        text = hv.Labels([(x, y, str(label))]).opts(
            text_font_size='10pt', 
            text_color=color
        )

        # text.style.fill = 'red'
        # Add the rectangle and its label to the overlay
        labelled_rects *= hv.Rectangles([rect]).opts(fill_color=None, line_color=color) * text
        # labelled_rects *= hv.DynamicMap(lambda: hv.Labels([text_label]))
    return labelled_rects

def download_dataset(dataset_name, file_id, destination_dir='dataset'):
    """
    Downloads a dataset file from Google Drive.

    Args:
        dataset_name (str): Name of the dataset.
        file_id (str): ID of the file on Google Drive.
        destination_dir (str, optional): Destination directory to save the file. Defaults to 'dataset'.
    """
    if not os.path.exists(destination_dir):
        os.mkdir(destination_dir)
    dataset_dir = os.path.join(destination_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    file_destination = os.path.join(dataset_dir, 'yolov5-thah.pt')
    subprocess.run(['gdown', f'https://drive.google.com/uc?id={file_id}', '-O', file_destination])
    

def megdown(fout_name, file_id, destination_dir):
    if not os.path.exists('datasets'):
        os.mkdir('datasets')
    destination_dir = 'datasets/' + destination_dir
    if not os.path.exists(destination_dir):
        os.mkdir(destination_dir)
    dataset_dir = os.path.join(destination_dir, fout_name)
    # if not os.path.exists(dataset_dir):
    #     os.mkdir(dataset_dir)

    # file_destination = os.path.join(dataset_dir, fout_name)
    # print(file_destination)
    subprocess.run(['gdown', f'https://drive.google.com/uc?id={file_id}', '-O', dataset_dir])