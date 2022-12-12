import pandas as pd 
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import cv2, glob, os, torch, math, torchvision, datetime
from pl_bolts.models.detection import YOLO, YOLOConfiguration
from tqdm.notebook import tqdm 
import utm

# my files
from mot_tracker.deep_sort import DeepSort
from mot_tracker.naive_sort import Sort
import draw_utils, transform_utils
from trajectory_extractor import TrajectoryExtractor

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    for detector_name in ['FasterRCNN']:
        for conf_thres in [0.01, 0.5, 0.9]: 
            for max_cos_dist in [0.05, 0.1, 0.2, 0.5]: 
                for max_iou_dist in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1]: 
                    for max_age in [1,3,5,7]:
                        #### args ###
                        class TrajectoryExtractorArgs: 
                            detector_name = detector_name
                            tracker_name = 'SORT'

                            unique_suffix = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')
                        #     input_video_path = "videos/Stadium/00084.wmv"
                        #     input_video_path = "videos/Y2E2/Y2E2_West.MOV"
                        #     input_video_path = "videos/GCS/grandcentral.avi" # cannot use because no gt. fml.
                            input_video_path = "videos/GCS/slideshow.avi" # Use this for paper YOLO. 
                        #     input_video_path = "videos/GCS/slideshow_small.avi" # Use this for paper FRCNN. 

                            output_txt_folder = 'TrackEval/data/trackers/mot_challenge/GCS-val/{}_{}_{}'.format(detector_name, tracker_name, unique_suffix)
                            if not os.path.exists(output_txt_folder):
                                os.makedirs(output_txt_folder)
                                os.makedirs(output_txt_folder+'/data')
                            output_txt_path = output_txt_folder+'/data/seq-01.txt'
                            args_save_txt_path = output_txt_folder+'/args.txt'
#                             output_video_path = os.path.split(input_video_path)[0]+"/out/{}.avi".format(unique_suffix)
                            output_video_path = "../../media/vivian/ExFAT-2TB/GCS/out/{}.avi".format(unique_suffix)

                        #     yolo_config_path = "yolo/yolov4-tiny-3l.cfg"
                        #     yolo_pre_weights_path = "yolo/yolov4-tiny.weights"
                            yolo_config_path = "yolo/yolov7-tiny.cfg"
                            yolo_pre_weights_path = "yolo/yolov7-tiny.weights"

                            deepsort_parameters_path = "deep_sort/deep/checkpoint/ckpt.t7"
                            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                            confidence_threshold = conf_thres
                            max_cos_dist = max_cos_dist
                            max_iou_dist = max_iou_dist
                            max_age = max_age

                        args = TrajectoryExtractorArgs

                        # set up our extractor
                        extractor = TrajectoryExtractor(args)

                        ##################################################
                        if args.detector_name == 'YOLO':
                            # set up yolo
                            yolo_config = YOLOConfiguration(args.yolo_config_path)
                            yolo_config.width = extractor.imw
                            yolo_config.height = extractor.imh
                            model = YOLO(network=yolo_config.get_network()).to(device)
                            with open(args.yolo_pre_weights_path) as pre_weights: 
                                model.load_darknet_weights(pre_weights)

                            extractor.load_detector(model)

                        ##################################################
                        elif args.detector_name == 'FasterRCNN':
                            # set up faster rcnn

                            # implemented to add the "infer" function to infer on single image 
                            anchor_generator = torchvision.models.detection.anchor_utils.AnchorGenerator(sizes=((8, 16, 32, 64, 128),),
                                                                                                         aspect_ratios=((0.5, 1.0),))
                            class modified_FasterRCNN(torch.nn.Module): 
                                def __init__(self, **kwargs):
                                    super().__init__()
                                    self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(**kwargs)
                                def infer(self, rgb_tensor): 
                                    # single image
                                    pred_dict = self.model([rgb_tensor])[0]
                                    boxes = pred_dict['boxes'] # (N,4)
                                    labels = pred_dict['labels'] # (N)
                                    labels = [l - 1 for l in labels] # subtract 1 from class IDs because coco classes have background=0. person=1
                                    confidences = pred_dict['scores'] # (N)  
                                    num_boxes = boxes.shape[0]

                                    return boxes, confidences, labels

                            model = modified_FasterRCNN(weights='DEFAULT').to(device)
                            extractor.load_detector(model)

                        ##################################################
                        if args.tracker_name == 'DeepSORT':
                            # set up deepsort
                            extractor.load_tracker(
                                DeepSort(args.deepsort_parameters_path, 
                                         min_confidence = args.confidence_threshold,
                                         max_dist = args.max_cos_dist, 
                                         max_iou_distance = args.max_iou_dist, 
                                         max_age = max_age,
                        #                  nms_max_overlap=1000
                                        )
                            )

                        elif args.tracker_name == 'SORT': 
                            # set up sort
                            extractor.load_tracker(
                                Sort(max_age=args.max_age, 
                                     min_hits=3,
                                     iou_threshold = 1- args.max_iou_dist)
                            )

                        detector_out_df, tracker_out_df, txt_df = extractor.detect_all(
                            max_frames=1000,
                            skip_every_n_frames = 1
                        )

                        extractor.release_all()