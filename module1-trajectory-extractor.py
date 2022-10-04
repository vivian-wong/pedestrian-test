#!/usr/bin/env python
# coding: utf-8

# # Import Packages

# In[2]:


# !wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny-3l.cfg
# !wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights


# In[3]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
import pandas as pd 
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import cv2, os, torch
from pl_bolts.models.detection import YOLO, YOLOConfiguration


# # Defined User Args

# In[4]:


class TrajectoryExtractorArgs: 
    video_path = "videos/Y2E2/Y2E2_West.MOV"
    yolo_config_path = "yolov4-tiny-3l.cfg"
    yolo_pre_weights_path = "yolov4-tiny.weights"
args = TrajectoryExtractorArgs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# # Step 1: DL model to track human locations in frames

# In[5]:


cap = cv2.VideoCapture(args.video_path)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = int(cap.get(cv2.CAP_PROP_FPS))
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


# In[6]:


ret, frame = cap.read()


# In[7]:


# rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# plt.imshow(rgb_img)


# In[8]:


# yolov4
yolo_config = YOLOConfiguration(args.yolo_config_path)
yolo_config.width = width
yolo_config.height = height
model = YOLO(network=yolo_config.get_network()).to(device)
with open(args.yolo_pre_weights_path) as pre_weights: 
    model.load_darknet_weights(pre_weights)


# In[1]:


model.infer(torch.rand([3,yolo_config.width,yolo_config.height]))


# In[7]:


# batch = torch.from_numpy(np.reshape(frame, (-1, 3, height, width)))


# # Step 2: Obtain 2D Plane Projection （TODO: automate this)

# In[ ]:





# # Step 3: Obtain the transformed human trajectories in world coordinates

# In[ ]:





# # Step 4: Identify crowdedness severity

# In[ ]:




