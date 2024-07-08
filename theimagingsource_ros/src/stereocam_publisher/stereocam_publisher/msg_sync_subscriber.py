import json
import yaml
import time
from datetime import datetime
import cv2
import sys
import os
import numpy as np
sys.path.append("../../tis_repos/Linux-tiscamera-Programming-Samples/python/python-common")
sys.path.append("../../theimagingsource_ros/config")
sys.path.append("/../../theimagingsource_ros/src/stereocam_publisher/stereocam_publisher")
from stereocam_publisher import TIS
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from stereo_msgs.msg import DisparityImage
from ament_index_python.packages import get_package_share_directory
import message_filters
from rclpy.qos import QoSProfile, qos_profile_sensor_data, HistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy

class CAMERA(TIS.TIS):
    
    def __init__(self, properties, imageprefix):
        '''
        Constructor of the CAMERA class
        :param properties: JSON object, that contains the list of to set properites
        :param triggerproperty: JSON object, which contains the trigger property name and the enable and disable values
        :param imageprefix: Used to create the file names of the images to be saved.
        '''
        super().__init__()
        self.properties = properties
        self.imageprefix = imageprefix
        self.busy = False
        self.imageCounter = 0
        

    def set_property(self, property_name, value):
        '''
        Pass a new value to a camera property. If something fails an
        exception is thrown.
        :param property_name: Name of the property to set
        :param value: Property value. Can be of type int, float, string and boolean
        '''
        try:
            baseproperty = self.source.get_tcam_property(property_name)
            baseproperty.set_value(value)
        except Exception as error:
            raise RuntimeError(f"Failed to set property '{property_name}'") from error
        
    def enableTriggerMode(self, onoff):
        '''
        Enable or disable the trigger mode
        :param bool onoff: "On" or "Off"
        '''       
        try:
            self.set_property("TriggerMode", onoff)
        except Exception as error:
            print(error)

    def applyProperties(self):
        '''
        Apply the properties in self.properties to the used camera
        The properties are applied in the sequence they are saved
        int the json file.
        Therefore, in order to set properties, that have automatiation
        the automation must be disabeld first, then the value can be set.
        '''
        for prop in self.properties:
            try:
                self.set_property(prop['property'],prop['value'])
            except Exception as error:
                print(error)       

def parameters_parsing(self):
        
    cam_serial = self.declare_parameter("serial", '')
    self.serial = cam_serial.get_parameter_value().string_value

    cam_prefix = self.declare_parameter("imageprefix", '')
    self.prefix = cam_prefix.get_parameter_value().string_value

    config_path = self.declare_parameter("config_path", '')
    self.config_path = config_path.get_parameter_value().string_value

    calib_path = self.declare_parameter("calib_path", '')
    self.calib_path = calib_path.get_parameter_value().string_value

    image_height = self.declare_parameter("image_height",'')
    self.image_height = image_height.get_parameter_value().string_value

    image_width = self.declare_parameter("image_width",'')
    self.image_width = image_width.get_parameter_value().string_value

    intrinsic_matrix = self.declare_parameter("intrinsic_matrix", rclpy.Parameter.Type.DOUBLE_ARRAY)
    self.intrinsic_matrix = intrinsic_matrix.get_parameter_value().double_array_value

    distortion_coefficients = self.declare_parameter("distortion_coefficients", rclpy.Parameter.Type.DOUBLE_ARRAY)
    self.distortion_coefficients = distortion_coefficients.get_parameter_value().double_array_value

    distortion_model = self.declare_parameter("distortion_model", '')
    self.distortion_model = distortion_model.get_parameter_value().string_value

    projection_matrix = self.declare_parameter("projection_matrix", rclpy.Parameter.Type.DOUBLE_ARRAY)
    self.projection_matrix = projection_matrix.get_parameter_value().double_array_value

    rectification_matrix = self.declare_parameter("rectification_matrix", rclpy.Parameter.Type.DOUBLE_ARRAY)
    self.rectification_matrix = rectification_matrix.get_parameter_value().double_array_value

    # with open(self.config_path) as jsonFile:
    #     self.prop = json.load(jsonFile)

    # self.format = self.prop["format"]
    # self.pformat = self.prop["pixelformat"]
    # self.width = self.prop["width"]
    # self.height = self.prop["height"]
    # self.framerate = self.prop["framerate"]
    
    print("node config:", self.prefix, self.pformat, self.serial, self.width, self.height, self.framerate)
    # print()


class Subscriber(Node):
    def __init__(self):
        super().__init__("image_camera_sync")
        # parameters_parsing(self)
        qos_policy = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST,
                                depth=5,
                                durability=QoSDurabilityPolicy.VOLATILE)

        # global ros_time 
        
        # image_raw_sub_l = message_filters.Subscriber(self, Image, "/stereo/left/image_raw")
        # image_raw_sub_r = message_filters.Subscriber(self, Image, "/stereo/right/image_raw")

        image_color_sub_l = message_filters.Subscriber(self, Image, "left/image_rect_color", qos_profile=qos_policy)
        image_color_sub_r = message_filters.Subscriber(self, Image, "right/image_rect_color", qos_profile=qos_policy)

        camera_info_sub_l = message_filters.Subscriber(self, CameraInfo, "left/camera_info", qos_profile=qos_policy)
        camera_info_sub_r = message_filters.Subscriber(self, CameraInfo, "right/camera_info", qos_profile=qos_policy)

        image_rect_l = message_filters.Subscriber(self, Image, "left/image_rect", qos_profile=qos_policy)
        image_rect_r = message_filters.Subscriber(self, Image, "right/image_rect", qos_profile=qos_policy)

        # disp = message_filters.Subscriber(self, DisparityImage, "/disparity")

        # sync_disp_l = message_filters.TimeSynchronizer(
        # [disp, image_rect_l],
        # queue_size=10)
        # sync_disp_r = message_filters.TimeSynchronizer(
        # [disp, image_rect_r],
        # queue_size=10)
        sync_l = message_filters.TimeSynchronizer(
        [image_rect_l, camera_info_sub_l],
        queue_size=9)
        # slop=0.1,
        # allow_headerless=False
        # )
        sync_r = message_filters.TimeSynchronizer(
        [image_rect_r, camera_info_sub_r],
        queue_size=9)
        # slop=0.1,
        # allow_headerless=False
        # )
        sync_caminfo = message_filters.TimeSynchronizer(
        [camera_info_sub_l, camera_info_sub_r],
        queue_size=9)
        # slop=0.1,
        # allow_headerless=False
        # )
        sync_color = message_filters.TimeSynchronizer(
        [image_color_sub_l, image_color_sub_r],
        queue_size=9)
        # slop=0.1,
        # allow_headerless=False
        # )
        sync_rect = message_filters.TimeSynchronizer(
        [image_rect_l, image_rect_r],
        queue_size=9)
        # slop=0.1,
        # allow_headerless=False,
        # )
        # sync_raw_l = message_filters.TimeSynchronizer(
        # [image_raw_sub_l, camera_info_sub_l],
        # queue_size=10)
        # slop=0.1,
        # allow_headerless=False,
        # )
        # sync_raw_r = message_filters.TimeSynchronizer(
        # [image_raw_sub_r, camera_info_sub_r],
        # queue_size=10)
        # slop=0.1,
        # allow_headerless=False,
        # )
        sync_color_l = message_filters.TimeSynchronizer(
        [image_color_sub_l, camera_info_sub_l],
        queue_size=9)
        # slop=0.1,
        # allow_headerless=False,
        # )
        sync_color_r = message_filters.TimeSynchronizer(
        [image_color_sub_r, camera_info_sub_r],
        queue_size=9)
        # slop=0.1,
        # allow_headerless=False,
        # )

        # sync_raw_l.registerCallback(raw_l_callback)
        # sync_raw_r.registerCallback(raw_r_callback)
        sync_color_l.registerCallback(color_l_callback)
        sync_color_r.registerCallback(color_r_callback)
        sync_color.registerCallback(sync_color_callback)
        sync_rect.registerCallback(sync_rect_callback)
        sync_l.registerCallback(sync_l_callback)
        sync_r.registerCallback(sync_r_callback)
        sync_caminfo.registerCallback(sync_caminfo_callback)
        # sync_disp_l.registerCallback(sync_disp_callback)
        # sync_disp_r.registerCallback(sync_disp_callback)

       
def sync_l_callback(image_rect_sub_l, camera_info_sub_l):
    ros_time = rclpy.clock.Clock().now().to_msg()
    image_rect_sub_l.header.stamp = ros_time
    camera_info_sub_l.header.stamp = ros_time
def sync_r_callback(image_rect_sub_r, camera_info_sub_r):
    ros_time = rclpy.clock.Clock().now().to_msg()
    image_rect_sub_r.header.stamp = ros_time
    camera_info_sub_r.header.stamp = ros_time
def sync_caminfo_callback(camera_info_sub_l, camera_info_sub_r):
    ros_time = rclpy.clock.Clock().now().to_msg()
    camera_info_sub_l.header.stamp = ros_time
    camera_info_sub_r.header.stamp = ros_time
# def raw_l_callback(image_raw_sub_l, camera_info_sub_l):
#     image_raw_sub_l.header.stamp = camera_info_sub_l.header.stamp
#     print('Sub raw l')
# def raw_r_callback(image_raw_sub_r, camera_info_sub_r):
#     image_raw_sub_r.header.stamp = camera_info_sub_r.header.stamp
#     print('Sub raw r')
# def mono_l_callback(image_mono_sub_l, camera_info_sub_l):
#     image_mono_sub_l.header.stamp = camera_info_sub_l.header.stamp
#     print('Sub mono l')
# def mono_r_callback(image_mono_sub_r, camera_info_sub_r): 
#     image_mono_sub_r.header.stamp = camera_info_sub_r.header.stamp
#     print('Sub mono r') 
def color_l_callback(image_color_sub_l, camera_info_sub_l):
    ros_time = rclpy.clock.Clock().now().to_msg()
    image_color_sub_l.header.stamp = ros_time
    camera_info_sub_l.header.stamp = ros_time
def color_r_callback(image_color_sub_r, camera_info_sub_r):
    ros_time = rclpy.clock.Clock().now().to_msg()
    image_color_sub_r.header.stamp = ros_time
    camera_info_sub_r.header.stamp = ros_time
def sync_rect_callback(image_rect_l, image_rect_r):
    ros_time = rclpy.clock.Clock().now().to_msg()
    image_rect_l.header.stamp = ros_time
    image_rect_r.header.stamp = ros_time
# def sync_mono_callback(image_mono_sub_l, image_mono_sub_r):
#     image_mono_sub_l.header.stamp = image_mono_sub_r.header.stamp
#     print('Sub sync mono')
def sync_color_callback(image_color_sub_l, image_color_sub_r):
    ros_time = rclpy.clock.Clock().now().to_msg()
    image_color_sub_l.header.stamp = ros_time
    image_color_sub_r.header.stamp = ros_time


class SyncChecker(Node):

    def __init__(self):
        super().__init__('sync_checker')
        self.sub_left = self.create_subscription(Image, '/left/image_raw', self.left_image_callback, 10)
        self.sub_right = self.create_subscription(Image, '/right/image_raw', self.right_image_callback, 10)
        self.left_time = None
        self.right_time = None

    def left_image_callback(self, msg):
        self.left_time = msg.header.stamp
        self.check_sync()

    def right_image_callback(self, msg):
        self.right_time = msg.header.stamp
        self.check_sync()

    def check_sync(self):
        if self.left_time is not None and self.right_time is not None:
            time_diff = (self.left_time - self.right_time).to_sec()
            if abs(time_diff) < 0.02:  
                self.get_logger().info('Stereo cameras are synchronized')
            else:
                self.get_logger().info('Stereo cameras are not synchronized')



def main(args=None):
    rclpy.init(args=args)

    stereocam_Subscriber = Subscriber()
    # stereocam_Synccheck = SyncChecker()
    rclpy.spin(stereocam_Subscriber)
    # rclpy.spin(stereocam_Synccheck)
    # stereocam_Subscriber.destroy_node()
    # stereocam_Synccheck.destroy_node()
    # rclpy.shutdown()

if __name__ == '__main__':
    main()

