
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, TextSubstitution

def generate_launch_description():
    ld = LaunchDescription()
    config = os.path.join(get_package_share_directory('stereocam_publisher'),
                          'config',
                          'nodes_config.yaml'
    )
    lcam_node = Node(
        package = "stereocam_publisher",
        executable = "leftcamera_publisher",
        # namespace = "stereo",
        name = "left",
        parameters = [config],
        # remappings = [
        #     ('/leftcam/left_image', '/left/image_raw'),
        #     ('/leftcam/left/camera_info', '/left/camera_info')
        # ]
    )
    rcam_node = Node(
        package = "stereocam_publisher",
        executable = "rightcamera_publisher",
        # namespace = "stereo",
        name = "right",
        parameters = [config],
        # remappings = [
        #     ('/rightcam/right_image', '/right/image_raw'),
        #     ('/rightcam/right/camera_info', '/right/camera_info')
        # ]
    )
    sync_node = Node(
        package = "stereocam_publisher",
        executable = "msg_sync_subscriber",
        # namespace = "stereo",
        name = "msg_sync_subscriber",
        
    )

    declare_approximate_sync = DeclareLaunchArgument(
        name='approximate_sync', default_value='False',
        description='Whether to use approximate synchronization of topics. Set to true if '
                    'the left and right cameras do not produce exactly synced timestamps.'
    )

    image_processing_launch_file_dir = os.path.join(get_package_share_directory('stereo_image_proc'), 'launch')

    image_processing_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(image_processing_launch_file_dir, 'stereo_image_proc.launch.py')
        ),
        launch_arguments={
            'approximate_sync': LaunchConfiguration('approximate_sync')
        }.items()
    )

    
    

    # disp_node = Node(
    #     package = "stereo_image_proc",
    #     executable = "disparity_node",
    #     namespace = "stereo",
    #     name = "disparity_node",
    #     arguments = ['_approximate_sync', 'True'],
    #     output = "screen",
    #     # remappings = [
    #     #     ('/stereo/right/image_raw', '/right/image_raw'),
    #     #     ('/stereo/right/camera_info', '/right/camera_info')
    #     # ]
    # )
    # pc_node = Node(
    #     package = "stereo_image_proc",
    #     executable = "point_cloud_node",
    #     namespace = "stereo",
    #     name = "point_cloud_node",
    #     arguments = ['_approximate_sync', 'True'],
    #     output = "screen",
    #     # remappings = [
    #     #     ('/rightcam/right_image', '/right/image_raw'),
    #     #     ('/rightcam/right/camera_info', '/right/camera_info')
    #     # ]
    # )
    
    ld.add_action(lcam_node)
    ld.add_action(rcam_node)
    # ld.add_action(sync_node)
    ld.add_action(declare_approximate_sync)
    ld.add_action(image_processing_cmd)
    # ld.add_action(disp_node)
    # ld.add_action(pc_node)

    return ld
