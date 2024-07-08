import numpy as np
import torch
import torch.nn.functional as F
# from PIL import Image
# import gradio as gr

from unimatch.unimatch import UniMatch
from utils.flow_viz import flow_to_image
from dataloader.stereo import transforms
from utils.visualization import vis_disparity
import calibration_configs

import cv2
import sys
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import message_filters
import math
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, HistoryPolicy, QoSDurabilityPolicy
import time
import os
import json
# import evaluate
from rclpy.executors import MultiThreadedExecutor
import threading

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

left_image_list, right_image_list = [], []

@torch.no_grad()
def inference(image1, image2, task='stereo'):
    """Inference on an image pair for optical flow or stereo disparity prediction"""

    model = UniMatch(feature_channels=128,
                     num_scales=2,
                     upsample_factor=4,
                     ffn_dim_expansion=4,
                     num_transformer_layers=6,
                     reg_refine=True,
                     task=task)
    
    # CPU is slow: 5s/frame
    model.to(device)
    model.eval()

    if task == 'flow':
        checkpoint_path = 'pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth'
    else:
        checkpoint_path = 'pretrained/gmstereo-scale2-regrefine3-resumeflowthings-mixdata-train320x640-ft640x960-e4e291fd.pth'

    checkpoint_flow = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint_flow['model'], strict=True)

    padding_factor = 32
    attn_type = 'swin' if task == 'flow' else 'self_swin2d_cross_swin1d'
    attn_splits_list = [2, 8]
    corr_radius_list = [-1, 4]
    prop_radius_list = [-1, 1]
    num_reg_refine = 6 if task == 'flow' else 3

    # smaller inference size for faster speed
    max_inference_size = [384, 768] if task == 'flow' else [640, 960]

    transpose_img = False

    # array <--> tensor CPU <--> GPU
    image1 = np.array(image1.detach().cpu().numpy()).astype(np.float32)
    image2 = np.array(image2.detach().cpu().numpy()).astype(np.float32)

    if len(image1.shape) == 2:  # gray image
        image1 = np.tile(image1[..., None], (1, 1, 3))
        image2 = np.tile(image2[..., None], (1, 1, 3))
    else:
        image1 = image1[..., :3]
        image2 = image2[..., :3]

    if task == 'flow':
        image1 = torch.from_numpy(image1).permute(2, 0, 1).float().unsqueeze(0)
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float().unsqueeze(0)
    else:
        val_transform_list = [transforms.ToTensor(),
                              transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                              ]

        val_transform = transforms.Compose(val_transform_list)

        sample = {'left': image1, 'right': image2}
        sample = val_transform(sample)

        image1 = sample['left'].unsqueeze(0)  # [1, 3, H, W]
        image2 = sample['right'].unsqueeze(0)  # [1, 3, H, W]

    # the model is trained with size: width > height
    if task == 'flow' and image1.size(-2) > image1.size(-1):
        image1 = torch.transpose(image1, -2, -1)
        image2 = torch.transpose(image2, -2, -1)
        transpose_img = True

    nearest_size = [int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,
                    int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor]

    inference_size = [min(max_inference_size[0], nearest_size[0]), min(max_inference_size[1], nearest_size[1])]

    assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
    ori_size = image1.shape[-2:]

    # resize before inference
    if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
        image1 = F.interpolate(image1, size=inference_size, mode='bilinear',
                               align_corners=True)
        image2 = F.interpolate(image2, size=inference_size, mode='bilinear',
                               align_corners=True)

    results_dict = model(image1.to(device), image2.to(device),
                         attn_type=attn_type,
                         attn_splits_list=attn_splits_list,
                         corr_radius_list=corr_radius_list,
                         prop_radius_list=prop_radius_list,
                         num_reg_refine=num_reg_refine,
                         task=task,
                         )

    flow_pr = results_dict['flow_preds'][-1]  # [1, 2, H, W] or [1, H, W]

    # resize back
    if task == 'flow':
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear',
                                    align_corners=True)
            flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
            flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]
    else:
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            pred_disp = F.interpolate(flow_pr.unsqueeze(1), size=ori_size,
                                      mode='bilinear',
                                      align_corners=True).squeeze(1)  # [1, H, W]
            pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

    if task == 'flow':
        if transpose_img:
            flow_pr = torch.transpose(flow_pr, -2, -1)

        flow = flow_pr[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]

        output = flow_to_image(flow)  # [H, W, 3]
    else:
        disp = pred_disp[0].cpu().numpy()

        output = vis_disparity(disp, return_rgb=True)

    return np.array(output)

class Unimatch_depth(Node):
    def __init__(self):
        super().__init__("Unimatch_depth")

        threading.Thread(target=self.depth_pub_callback).start()

        self.bridge = CvBridge()
        self.i = 1
        qos_policy = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_ALL,
                                depth=20,
                                durability=QoSDurabilityPolicy.VOLATILE)
        
        self.left_sub = message_filters.Subscriber(self, Image, "/left/image_raw", qos_profile=qos_policy)
        self.right_sub = message_filters.Subscriber(self, Image, "/right/image_raw", qos_profile=qos_policy)

        sync_raw = message_filters.ApproximateTimeSynchronizer(
        [self.left_sub, self.right_sub],
        queue_size=200,
        slop=0.05,
        allow_headerless=False)
        sync_raw.registerCallback(self.sync_raw_callback)

        self.depth_publish = self.create_publisher(Image, 'left/image_depth', 20)
        self.timer = self.create_timer(0.1, self.depth_pub_callback)
    
    def sync_raw_callback(self, imgl_raw_suber, imgr_raw_suber):
        ros_time = rclpy.clock.Clock().now().to_msg()
        imgl_raw_suber.header.stamp = ros_time
        imgr_raw_suber.header.stamp = ros_time
      
        if imgl_raw_suber.header.frame_id == "left_camera":
            self.left_raw = self.bridge.imgmsg_to_cv2(imgl_raw_suber, "mono8")
            left_image_list.append(self.left_raw)
            self.get_logger().info(f'Received_left_image: {self.i}')
            
        if imgr_raw_suber.header.frame_id == "right_camera":
            self.right_raw = self.bridge.imgmsg_to_cv2(imgr_raw_suber, "mono8")
            right_image_list.append(self.right_raw)
            self.get_logger().info(f'Received_right_image: {self.i}')
        
        self.i += 1

    def depth_pub_callback(self):
        
        time1 = time.time()
        fps = 0.0

        for self.left_image_raw, self.right_image_raw in zip(left_image_list, right_image_list):

            # .to(device) sonly operates in place when applied to a model. When applied to a tensor, it must be assigned:
            self.left_image_raw = torch.from_numpy(self.left_image_raw).to(device) # .cuda() 
            self.right_image_raw = torch.from_numpy(self.right_image_raw).to(device) # .cuda()
            # print(type(self.left_image_raw), device)
            
            img_depth = inference(self.left_image_raw, self.right_image_raw)
            ##########################
            img_depth_gray = cv2.cvtColor(img_depth, cv2.COLOR_BGR2GRAY)
            threeD = cv2.reprojectImageTo3D(img_depth_gray.astype(np.float32), calibration_configs.Q, handleMissingValues=True)
            threeD = threeD  / 16
            x1, y1 = 300, 300
            distance1 = math.sqrt(threeD[y1][x1][0]**2 + threeD[y1][x1][1]**2 + threeD[y1][x1][2]**2)
            distance1 = distance1 / 1000.0
            img_depth = cv2.rectangle(img_depth, (260,340), (340,260), color=(255,255,255), thickness=2)
            img_depth = cv2.putText(img_depth, "Depth: %.2fm" % (distance1), (160, 200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=2)
            ##############################

            time2 = time.time()
            infer_time = time2 - time1
            fps += (1./infer_time)

            img_depth = cv2.putText(img_depth, "InferenceTime = %.2fs/frame" % (infer_time), org=(0, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255,255,255), thickness=2)
            img_depth = cv2.putText(img_depth, "FPS = %.2f" % (fps), org=(0, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255,255,255), thickness=2)
            cv2.imshow("Unimatch_depth", img_depth)
            cv2.waitKey(1)
            
            depth_msg = self.bridge.cv2_to_imgmsg(np.array(img_depth), "8UC3")
            ros_time_msg = rclpy.clock.Clock().now().to_msg()
            depth_msg.header.frame_id = "left_camera"
            depth_msg.header.stamp = ros_time_msg

            self.depth_publish.publish(depth_msg)

            left_image_list.clear()
            right_image_list.clear()
    


def main(args=None):
    rclpy.init(args=args)
    try:
        depth_publisher = Unimatch_depth()
        executor = MultiThreadedExecutor()
        executor.add_node(depth_publisher)

        try:
            executor.spin()
        finally:
            executor.shutdown()
            depth_publisher.destroy_node()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()

