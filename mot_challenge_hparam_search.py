import pandas as pd 
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import cv2, os, torch, math, torchvision, datetime
from pl_bolts.models.detection import YOLO, YOLOConfiguration, FasterRCNN
from tqdm import tqdm 
import utm

# my files
from deep_sort.deep_sort import DeepSort
from trajectory_extractor import TrajectoryExtractor

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)


    for conf_thres in [0.01, 0.1, 0.2, 0.3, 0.5, 0.7]: 
        for max_cos_dist in [0.05, 0.1, 0.2, 0.5]: 
            for max_iou_dist in [0.3, 0.5, 0.7]: 
                print('################')
                print('conf_thres, max_cos_dist, max_iou_dist')
                print(conf_thres, max_cos_dist, max_iou_dist)
                #### args ###
                class TrajectoryExtractorArgs: 
                    unique_suffix = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')
                #     input_video_path = "videos/Stadium/00084.wmv"
                #     input_video_path = "videos/Y2E2/Y2E2_West.MOV"
                #     input_video_path = "videos/GCS/grandcentral.avi"
                    MOT_subset_name = 'MOT20-05'
                    input_video_path = "videos/MOT20/train/{}/img1/%06d.jpg".format(MOT_subset_name)
                #     output_txt_path = output_video_path.replace("avi", "txt")

                    output_txt_folder = 'TrackEval/data/trackers/mot_challenge/MOT20-train/YOLO_DeepSORT_{}'.format(unique_suffix)
                    if not os.path.exists(output_txt_folder):
                        os.makedirs(output_txt_folder)
                        os.makedirs(output_txt_folder+'/data')
                    output_txt_path = output_txt_folder+'/data/{}.txt'.format(MOT_subset_name)
                    output_video_path = os.path.split(input_video_path)[0]+"/out_{}.avi".format(unique_suffix)

                #     yolo_config_path = "yolo/yolov4-tiny-3l.cfg"
                #     yolo_pre_weights_path = "yolo/yolov4-tiny.weights"
                    yolo_config_path = "yolo/yolov7-tiny.cfg"
                    yolo_pre_weights_path = "yolo/yolov7-tiny.weights"

                    confidence_threshold = 0.01#0.7
                    deepsort_parameters_path = "deep_sort/deep/checkpoint/ckpt.t7"
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


                args = TrajectoryExtractorArgs
                args.confidence_threshold = conf_thres
                print(args.unique_suffix)
                # set up our extractor
                extractor = TrajectoryExtractor(args)

                # set up yolo
                yolo_config = YOLOConfiguration(args.yolo_config_path)
                yolo_config.width = extractor.imw
                yolo_config.height = extractor.imh
                model = YOLO(network=yolo_config.get_network()).to(device)
                with open(args.yolo_pre_weights_path) as pre_weights: 
                    model.load_darknet_weights(pre_weights)

                extractor.load_detector(model)

                # # set up faster rcnn TODO
                # model = FasterRCNN(pretrained=True).to(device)
                # extractor.load_detector(model)

                # set up deepsort
                extractor.load_tracker(
                    DeepSort(args.deepsort_parameters_path, 
                             min_confidence = args.confidence_threshold, 
                             max_dist=max_cos_dist,
                             max_iou_distance=max_iou_dist)
                )

                detector_out_df, tracker_out_df, txt_df = extractor.detect_all(
                #     max_frames=100,
                    skip_every_n_frames = 1
                )

                extractor.release_all()