import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch
from PIL import Image

import utils.util as util
import data.util as data_util
from models import create_model

import os.path as osp
import logging
import argparse
from collections import OrderedDict
from models.hrseg_model import create_hrnet

import options.options as option
import utils.util as util

from models import create_model

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import time
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from stereo_msgs.msg import DisparityImage
from ament_index_python.packages import get_package_share_directory
import message_filters
import math
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, HistoryPolicy, QoSDurabilityPolicy
import time
import os

# torch.cuda.empty_cache()
print(torch.cuda.is_available())
print(torch.cuda.device_count())

# Arguments options
parser = argparse.ArgumentParser()
# parser.add_argument('-opt', default="options/test/LOLv1_seg.yml", type=str, help='Path to options YMAL file.')
parser.add_argument('-opt', default="./options/test/LOLv2_SKF.yml", type=str, help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

resize_image_list = []

class SNR_SKF(Node):
    def __init__(self):
        super().__init__("SNR_SKF")

        self.bridge = CvBridge()
        self.i = 1
        qos_policy = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_ALL,
                                depth=2000,
                                durability=QoSDurabilityPolicy.VOLATILE)
        # self.resize_sub = self.create_subscription(Image, 'left/image_resize', self.resize_sub_callback, qos_profile=qos_policy)
        self.seg_sub = self.create_subscription(Image, 'left/image_seg', self.resize_sub_callback, qos_profile=qos_policy)
        self.snr_publish = self.create_publisher(Image, 'left/image_snr', 2000)
        self.timer = self.create_timer(0.2, self.snr_pub_callback)
    
    def resize_sub_callback(self, resize_msg):
        
        self.LQ = self.bridge.imgmsg_to_cv2(resize_msg, 'bgr8')
        resize_image_list.append(self.LQ.astype(np.float32) / 255.)

        torch.cuda.empty_cache()
        self.get_logger().info(f'Received_resize_image: {self.i}')
        self.get_logger().info(f'Resize_timestamp: {resize_msg.header.stamp}')
        self.i += 1

    
    def snr_pub_callback(self):

        torch.cuda.empty_cache()
        if opt['seg']:
            seg_model = create_hrnet().cuda()
            seg_model.eval()
        else:
            seg_model = None
        model = create_model(opt)

        time1 = time.time()
        fps = 0.0

        for n in range(len(resize_image_list)):
            if n % 2:
                LQ = np.stack(resize_image_list[:n])[:, :, :, [2, 1, 0]]
                LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(LQ, (0, 3, 1, 2)))).float().cuda()
                nf = LQ[0,:,:,:].permute(1, 2, 0) * 255.0 # Tensor.permute, np.transpose
                nf = nf.detach().cpu()
                nf = cv2.blur(np.float32(nf), (5, 5))
                nf = nf * 1.0 / 255.0
                nf = torch.Tensor(nf).float().permute(2, 0, 1).cuda()
            
               
            
                if seg_model is not None:
                    seg_map, seg_feature = seg_model(LQ)
                        
                else:
                    seg_map, seg_feature = None, None
            
                model.feed_data_2(LQ, nf, seg_map, seg_feature)
                model.test()
                visuals = model.get_current_visuals()
                rlt_img = util.tensor2img(visuals['rlt'])  # uint8
                

                time2 = time.time()
                infer_time = time2 - time1
                fps = fps + (1. / infer_time)

                torch.cuda.empty_cache()

                # ROS msgs
                snr_msg = self.bridge.cv2_to_imgmsg(np.array(rlt_img), 'bgr8')
                ros_time = rclpy.clock.Clock().now().to_msg()
                snr_msg.header.frame_id = "left_camera"
                snr_msg.header.stamp = ros_time

                self.snr_publish.publish(snr_msg)

                del resize_image_list[:n]
            
            

def main(args=None):
    rclpy.init(args=args)
    snr_publisher = SNR_SKF()
    rclpy.spin(snr_publisher)
    snr_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
