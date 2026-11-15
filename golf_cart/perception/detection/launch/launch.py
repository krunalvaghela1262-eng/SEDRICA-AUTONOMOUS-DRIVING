import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg_name = 'detection'
    pkg_share_dir = get_package_share_directory(pkg_name)

    depthai_ros_driver_pkg = get_package_share_directory('depthai_ros_driver')
    camera_launch_file = os.path.join(depthai_ros_driver_pkg, 'launch', 'camera.launch.py')

    params_file = os.path.join(pkg_share_dir, 'config', 'params.yaml')

    oak_driver_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(camera_launch_file),
    )

    oak_driver_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(camera_launch_file),
    )
    
    detection_logic_node = Node(
        package=pkg_name,
        executable='detection_node',
        name='detection_logic_node',
        output='screen',
        parameters=[params_file]
    )

    return LaunchDescription([
        oak_driver_node,
        detection_logic_node,
    ])