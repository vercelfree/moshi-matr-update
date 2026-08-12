# -*- coding: utf-8 -*-
import sys
sys.path.append('./Evaluation')
from eval_detection_gentime import ANETdetection
import matplotlib.pyplot as plt
import numpy as np

def run_evaluation_detection(ground_truth_filename, prediction_filename,
                             blocked_videos, 
                             tiou_thresholds=np.linspace(0.5, 0.95, 10),
                             subset='validation', verbose=True):

    anet_detection = ANETdetection(ground_truth_filename, prediction_filename,
                                   subset=subset, tiou_thresholds=tiou_thresholds,
                                   verbose=verbose, check_status=False, blocked_videos=blocked_videos)
    anet_detection.evaluate()
    
    ap = anet_detection.ap
    mAP = anet_detection.mAP
    tdiff = anet_detection.tdiff
    
    return (mAP, ap, tdiff)

def evaluation_detection(args, output_file, subset, tiou_thresholds, verbose=True, blocked_videos=list()):
    if args.dataset == 'thumos14':
        gt_file_name = args.video_anno
        json_subset = subset
    elif args.dataset == "muses":
        gt_file_name = args.video_anno_muses
        json_subset = subset
    elif args.dataset == "epic":
        gt_file_name = args.video_anno
        json_subset = {'train': 'training', 'test': 'validation'}[subset]
    else:
        raise ValueError(f"Unknown args.dataset: {args.dataset!r}")
    mAP, AP, tdiff = run_evaluation_detection(
        gt_file_name,
        output_file,
        tiou_thresholds=np.linspace(0.3, 0.70, 5),
        subset=json_subset, verbose=verbose, blocked_videos=blocked_videos)
    mAP, AP, tdiff = mAP * 100, AP * 100, tdiff * 100
    if verbose:    
        print('mAP')
        print(mAP)
        print('AP')
        print(AP)
        print('AEDT')
        print(tdiff)
    
    return mAP