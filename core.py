from typing import Tuple
from numpy.typing import ArrayLike, NDArray
import numpy as np
from general import *
from scipy.optimize import linear_sum_assignment

def matching_iou_hungarian(iou_matrix, iou_threshold=0):
    iou_matrix = np.where(iou_matrix < iou_threshold, 0, iou_matrix)
    cost_matrix = 1 - iou_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind

def gather_nonconformity_scores(
        labels: ArrayLike,
        pred_prob: NDArray,
        method: str,
        k_reg=None,
        lambda_reg=None
    ) -> Tuple[NDArray, NDArray]:
        if method not in ['lac', 'aps', 'raps']:
            raise NotImplementedError("Expect method to be one of these: ['lac', 'aps', 'raps'].")

        labels = labels.copy()
        pred_prob = pred_prob.copy()
        if method == 'lac':
            # lac from "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification"
            # https://arxiv.org/abs/2107.07511
            nonconformity_scores = pred_prob[range(len(labels)), labels.argmax(axis=1)]
        elif method == 'aps':
            # aps from MAPIE: https://github.com/scikit-learn-contrib/MAPIE/blob/master/mapie/classification.py
            sorted_pi = np.fliplr(np.argsort(pred_prob, axis=1))
            pred_prob_sorted = np.take_along_axis(pred_prob, sorted_pi, axis=1)
            labels_sorted = np.take_along_axis(labels, sorted_pi, axis=1)
            pred_prob_cumsum = np.cumsum(pred_prob_sorted, axis=1)
            cutoff = np.argmax(labels_sorted, axis=1)
            nonconformity_scores = np.take_along_axis(pred_prob_cumsum, cutoff.reshape(-1, 1), axis=1)
        elif method == 'raps':
            # MAPIE and
            # https://arxiv.org/abs/2009.14193  : Uncertainty Sets for Image Classifiers using Conformal Prediction
            # https://arxiv.org/abs/2107.07511  : A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification
            # idk why MAPIE raps implementation is so unreadable for me, personally.
            if k_reg is None or lambda_reg is None:
                raise ValueError("k_reg and lambda_reg is require when using 'raps' method")
            reg_vec = np.zeros(pred_prob.shape[1])
            reg_vec[k_reg:] = lambda_reg
            
            index_sorted = np.fliplr(np.argsort(pred_prob, axis=1))
            y_pred_srt = np.take_along_axis(pred_prob, index_sorted, axis=1)
            y_pred_srt_reg = y_pred_srt + reg_vec
            cal_L = np.where(index_sorted == np.argmax(labels, axis=1)[:, None])
            nonconformity_scores = np.cumsum(y_pred_srt_reg,axis=1)[np.where(index_sorted == np.argmax(labels, axis=1)[:, None])]
            
            # nonconformity_scores = (
            #     np.cumsum(y_pred_srt_reg, axis=1)[np.arange(len(labels)), cal_L] 
            #     - np.random.rand(len(labels)) * y_pred_srt_reg[np.arange(len(labels)), cal_L]
            #     )
                
        return nonconformity_scores
    
def form_prediction_set(pred_dist, quantile, method, k_reg=None, lambda_reg=None, disallow_zero_sets=True):
    if method not in ['lac', 'aps', 'raps']:
        raise NotImplementedError("Expect method to be one of these: ['lac', 'aps', 'raps'].")
    
    if method == 'lac':
        prediction_set = pred_dist > quantile
    elif method == 'aps':
        sorted_index = np.argsort(pred_dist, axis=1)[:, ::-1]
        sorted_dist = np.take_along_axis(pred_dist, sorted_index, axis=1)
        cumsum_dist = np.cumsum(sorted_dist, axis=1)
        
        includeds = cumsum_dist <= quantile
        num_classes = pred_dist.shape[1]
        prediction_set = []
        
        assert np.equal(len(pred_dist), len(includeds)).all()

        for i, included in enumerate(includeds): # for each bounding box
            class_included = np.zeros(num_classes, dtype=bool)
            class_included[sorted_index[i, included]] = True
            prediction_set.append(class_included)

        prediction_set = np.array(prediction_set)
        # if disallow_zero_sets: indicators[:,0] = True
        try:
            prediction_set = np.vstack(prediction_set)
        except:
            pass
    elif method == 'raps':
        if k_reg is None or lambda_reg is None:
            raise ValueError("k_reg and lambda_reg is require when using 'raps' method")
        reg_vec = np.zeros(pred_dist.shape[1])
        reg_vec[k_reg:] = lambda_reg
        sorted_index = np.argsort(pred_dist, axis=1)[:, ::-1]
        sorted_dist = np.take_along_axis(pred_dist, sorted_index, axis=1)
        sorted_dist_reg = sorted_dist + reg_vec
        cumsum_dist = np.cumsum(sorted_dist_reg, axis=1)
        indicators = (cumsum_dist - np.random.rand(len(pred_dist),1)*sorted_dist_reg) <= quantile 
        if disallow_zero_sets: indicators[:,0] = True
        prediction_set = np.take_along_axis(indicators, sorted_index.argsort(axis=1),axis=1)
    return prediction_set




