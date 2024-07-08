
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
    
    
    rcam_node = Node(
        package = "stereocam_publisher",
        executable = "rightcamera_publisher",
        namespace = "rightcam",
        name = "rightcamera_publisher",
        parameters = [config]
    )

    

    ld.add_action(rcam_node)
    return ld
