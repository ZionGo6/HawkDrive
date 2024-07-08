from transformers import DPTForDepthEstimation, DPTFeatureExtractor
# from transformers import AutoImageProcessor, DPTForSemanticSegmentation
import torch
# from datasets import load_dataset
# from PIL import Image
import numpy as np
import cv2
import sys
# from PIL import Image
# import matplotlib.pyplot as plt
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
# from stereo_msgs.msg import DisparityImage
# from ament_index_python.packages import get_package_share_directory
import message_filters
import math
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, HistoryPolicy, QoSDurabilityPolicy
import time
import os
import json
from huggingface_hub import hf_hub_download
from torch import nn

from rclpy.executors import MultiThreadedExecutor
import threading


torch.cuda.empty_cache()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_folder = "/workspaces/isaac_ros-dev/SNR-aware/dpt-hybrid-midas"
model = DPTForDepthEstimation.from_pretrained(model_folder, low_cpu_mem_usage=True)
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
model.to(device)


image_list = []

# set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

def ade_palette():
    """ADE20K palette that maps each class to RGB values."""

    return [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255]]

def cityscapes_palette():
    '''
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    '''
    # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    return [[128, 64,128],[244, 35,232],[70, 70, 70],[102,102,156],
            [190,153,153],[153,153,153],[250,170, 30],
            [220,220,  0],[107,142, 35],[152,251,152],[70,130,180],[220, 20, 60],[255,  0,  0],
            [0,  0,142],[0,  0, 70],[0, 60,100],[0, 80,100],[0,  0,230],
            [119, 11, 32]]

class DPT(Node):
    def __init__(self):
        super().__init__("DPT")

        threading.Thread(target=self.dpt_pub_callback).start()

        self.bridge = CvBridge()
        self.i = 1
        qos_policy = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_ALL,
                                depth=20,
                                durability=QoSDurabilityPolicy.VOLATILE)
        
       
        self.dpt_sub = self.create_subscription(Image, 'left/image_raw', self.raw_sub_callback, qos_profile=qos_policy)
        self.dpt_publish = self.create_publisher(Image, 'left/image_dpt', 20)
        self.timer = self.create_timer(0.2, self.dpt_pub_callback)
    
    def raw_sub_callback(self, raw_msg):
        self.get_logger().info(f'Processed_resize_image: {self.i}')
        self.get_logger().info(f'DPT_timestamp: {raw_msg.header.stamp}')
        self.image = self.bridge.imgmsg_to_cv2(raw_msg, 'bgr8')
        
        image_list.append(self.image)
        torch.cuda.empty_cache()
        self.i += 1

    def dpt_pub_callback(self):
        
        time1 = time.time()
        fps = 0.0

        for n in range(len(image_list) + 1):
            if len(image_list) == 0:
                continue
            
            # input
            pixel_values = feature_extractor(image_list[n], return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                
                outputs = model(pixel_values) # DPTForDepthEstimation object argument after ** must be a mapping, not Tensor
                predicted_depth = outputs.predicted_depth
                
                # rescale logits to original image size
                prediction = torch.nn.functional.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=image_list[n].shape[0:2],
                    mode="bicubic",
                    align_corners=False,
                )
                
              
                
                torch.cuda.empty_cache()
                
                output = prediction.squeeze().cpu().numpy()
                formatted = (output * 255 / np.max(output)).astype("uint8")
                # depth = Image.fromarray(formatted)
                
                img_dpt = formatted.astype(np.uint8)
                
                time2 = time.time()
                infer_time = time2 - time1
                
                # calcualte the FPS
                fps = fps + (1. / infer_time)

                img_dpt = cv2.putText(img_dpt, "FPS = %.2f" % (fps), org=(0, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 255, 255), thickness=1)
                img_dpt = cv2.putText(img_dpt, "Inference Time = %.2fs/frame" % (infer_time), org=(0, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 255, 255), thickness=1)

                image_list.clear()

                # publish the images
                dpt_msg = self.bridge.cv2_to_imgmsg(np.array(img_dpt), '8UC1')
                ros_time_msg = rclpy.clock.Clock().now().to_msg()
                dpt_msg.header.frame_id = "left_camera"
                dpt_msg.header.stamp = ros_time_msg

                self.dpt_publish.publish(dpt_msg)

def main(args=None):
    rclpy.init(args=args)
    try:
        dpt_publisher = DPT()
        executor = MultiThreadedExecutor()
        executor.add_node(dpt_publisher)

        try:
            executor.spin()
            # torch.cuda.empty_cache()
        finally:
            executor.shutdown()
            dpt_publisher.destroy_node()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
    