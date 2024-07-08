
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
from launch.actions import DeclareLaunchArgument
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
        namespace = "leftcam",
        name = "leftcamera_publisher",
        parameters = [config]
    )
    
    

    ld.add_action(lcam_node)
    
    return ld
