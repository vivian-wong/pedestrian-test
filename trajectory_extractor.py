import pandas as pd 
import numpy as np
import cv2
from tqdm import tqdm
import transform_utils, draw_utils

class TrajectoryExtractor: 
    def __init__(self, 
                 args
                ): 
        self.args = args
        
        self.__load_video()
                
    def __load_video(self):
        print("Loading video from {}...".format(self.args.input_video_path))
        self.cap = cv2.VideoCapture(self.args.input_video_path)
        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.FPS = int(self.cap.get(cv2.CAP_PROP_FPS))
        print(self.FPS)
#         if self.FPS == 1.0: 
#             # images, use FPS=20 per mot challenge and GCS. 
#             self.FPS = 25
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_writer=None
        print("Loaded video has {} frames at FPS {}, size {}x{}".format(self.num_frames, 
                                                                        self.FPS, 
                                                                        self.width, 
                                                                        self.height))
        # find padded width and height
        random_rgb_tensor = transform_utils.pad_rgbarray_to_tensor(np.random.rand(self.height, self.width, 3),
                                               multiple=32)
        self.imw = random_rgb_tensor.shape[2]
        self.imh = random_rgb_tensor.shape[1]        
        print("transformed to size {}x{}".format(self.imw, self.imh))
    
    def load_detector(self, model): 
        model.eval()
        self.detector = model
    
    def load_tracker(self, mot_tracker): 
        self.tracker = mot_tracker
    
    def detect_all(self, 
                   max_frames = None, 
                   skip_every_n_frames = 1,
                   txt_format = 'modified_mot_challenge_2d_box_to_point'): 
        assert skip_every_n_frames >=1
        self.video_writer = cv2.VideoWriter(self.args.output_video_path,
                                            cv2.VideoWriter_fourcc(*'MJPG'),
                                            self.FPS/skip_every_n_frames, 
                                            (self.width, self.height))
        curr_frame_number = 0
        detector_outputs_df = pd.DataFrame()
        tracker_outputs_df = pd.DataFrame()
        if not max_frames: 
            max_frames = self.num_frames
        for ii in tqdm(range(max_frames)): 
            ret, frame = self.cap.read()
            if ret:
                detector_outputs, tracker_outputs = self.detect_frame(frame)
                detector_outputs['frame_number'] = ii+1 # frame number needs one indexed for mots challenge format
                detector_outputs_df = pd.concat([detector_outputs_df, pd.DataFrame.from_dict(detector_outputs)])
                if tracker_outputs:
                    tracker_outputs['frame_number'] = ii+1
                    tracker_outputs_df = pd.concat([tracker_outputs_df, pd.DataFrame.from_dict(tracker_outputs)])

                    # draw bounding boxes from tracker_outputs
#                     draw_utils.draw_boxes(   frame, 
#                                              ii,
#                                              detector_outputs['bbox_x1'],
#                                              detector_outputs['bbox_y1'],
#                                              detector_outputs['bbox_x2'],
#                                              detector_outputs['bbox_y2'],
#                                              [1]*len(detector_outputs['bbox_x1'])
#                                            )
                    draw_utils.draw_boxes(   frame, 
                                             ii,
                                             tracker_outputs['bbox_x1'],
                                             tracker_outputs['bbox_y1'],
                                             tracker_outputs['bbox_x2'],
                                             tracker_outputs['bbox_y2'],
                                             tracker_outputs['person_id']
                                           )
                curr_frame_number += skip_every_n_frames
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame_number)
                # save new video 
                self.video_writer.write(frame)
        
        if txt_format == 'original_mot_challenge_2d_box':
            # write results in mot format (bounding boxes)
            txt_df = tracker_outputs_df.copy()
            txt_df['bbox_w'] = txt_df['bbox_x2'] - txt_df['bbox_x1']
            txt_df['bbox_h'] = txt_df['bbox_y2'] - txt_df['bbox_y1']
            txt_df['conf'] = 1
            txt_df['x'] = -1
            txt_df['y'] = -1
            txt_df['z'] = -1                
            
        elif txt_format == 'modified_mot_challenge_2d_box_to_point':
            # write results in mot format (coordinates only)
            txt_df = tracker_outputs_df.copy()
            txt_df['bbox_x1'] = (txt_df['bbox_x1'] + txt_df['bbox_x2'])//2  # technically xc
            txt_df['bbox_y1'] = (txt_df['bbox_y1'] + txt_df['bbox_y2'])//2  # technically yc
            txt_df['bbox_w'] = 0
            txt_df['bbox_h'] = 0
            txt_df['conf'] = 1
            txt_df['x'] = -1
            txt_df['y'] = -1
            txt_df['z'] = -1    
        
        else: 
            assert False
            
        txt_df[['frame_number', 'person_id', 'bbox_x1', 'bbox_y1', 'bbox_w', 'bbox_h', 'conf', 'x', 'y', 'z']].to_csv(self.args.output_txt_path, 
                                                                                                                  header=None, 
                                                                                                                  index=None, 
                                                                                                                  sep=',', 
                                                                                                                  mode='w')
        pd.DataFrame.from_dict(self.args.__dict__, orient = 'index').to_csv(self.args.args_save_txt_path)

        return detector_outputs_df, tracker_outputs_df, txt_df
        
    def detect_frame(self, frame):
#         print(frame.shape)
        detector_outputs = {
            'crowd_count': [],
            'confidence': [],
            'bbox_x1': [],
            'bbox_y1': [],
            'bbox_x2': [], 
            'bbox_y2': [],
            'bbox_xc': [],
            'bbox_yc': [],
            'bbox_w': [], 
            'bbox_h': [],
            'cls_id': [],
        }
        tracker_outputs = dict()
        
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_tensor = transform_utils.pad_rgbarray_to_tensor(rgb_img, 32)
        frame = transform_utils.tensor_to_rgbarray(rgb_tensor)
        rgb_tensor = rgb_tensor.float().to(self.args.device)
        boxes, confidences, labels = self.detector.infer(rgb_tensor)
        cnt = 0
        for obj_id in range(boxes.shape[0]): 
            conf = round(confidences[obj_id].detach().cpu().item(),2)
            if (labels[obj_id] == 0) and (conf > self.args.confidence_threshold): # person
                x1, y1, x2, y2 = boxes[obj_id,:].detach().cpu().numpy().astype('int')
                assert x1 >= 0 and x2 >= 0 and y1 >= 0 and y2 >= 0
                # store detector_outputsults
                detector_outputs['crowd_count'].append(cnt)
                detector_outputs['confidence'].append(conf)
                # account for padding here since we're drawing and evaluating on original frame size. 
                x_pad_single_side = int((self.imw - self.width)/2)
                y_pad_single_side = int((self.imh - self.height)/2)
                detector_outputs['bbox_x1'].append(x1+x_pad_single_side)
                detector_outputs['bbox_y1'].append(y1+y_pad_single_side)
                detector_outputs['bbox_x2'].append(x2+x_pad_single_side)
                detector_outputs['bbox_y2'].append(y2+y_pad_single_side)
                detector_outputs['bbox_xc'].append((x1+x2)/2)
                detector_outputs['bbox_yc'].append((y1+y2)/2)
                detector_outputs['bbox_w'].append(x2-x1)
                detector_outputs['bbox_h'].append(y2-y1)
                detector_outputs['cls_id'].append(0)
                
                cnt += 1
        # deepsort
        bbox_xywh = np.array(list(zip(detector_outputs['bbox_xc'],
                     detector_outputs['bbox_yc'], 
                     detector_outputs['bbox_w'], 
                     detector_outputs['bbox_h'])))
        outputs = self.tracker.update(bbox_xywh, detector_outputs['confidence'], frame, detector_outputs['cls_id'])
#         outputs = self.tracker.update(bbox_xywh, [1]*len(detector_outputs['confidence']), frame, detector_outputs['cls_id'])
#         print('DEEPSORT', output)
        if len(outputs)>0:
            tracker_outputs['bbox_x1'] = outputs[:,0]
            tracker_outputs['bbox_y1'] = outputs[:,1]
            tracker_outputs['bbox_x2'] = outputs[:,2]
            tracker_outputs['bbox_y2'] = outputs[:,3]
            tracker_outputs['person_id'] = outputs[:,4]
            tracker_outputs['crowd_count'] = outputs.shape[0]
            
        
        return detector_outputs, tracker_outputs
    
    def release_all(self): 
        ''' run this after all codes'''
        self.cap.release()
        if self.video_writer:
            self.video_writer.release()
            
        cv2.destroyAllWindows()