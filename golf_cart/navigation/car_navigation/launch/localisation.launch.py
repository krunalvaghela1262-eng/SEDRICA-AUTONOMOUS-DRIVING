import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    
    # 1. Path to your specific map file
    map_file = '/home/jetson/Developer/navigation/src/car_navigation/map/front_lidar.yaml'

    # 2. Path to your params file (nav2_params.yaml)
    # Ensure this file actually exists at this location!
    params_file = '/home/jetson/Developer/navigation/src/car_navigation/config/gahan_params.yaml'

    # 3. Locate the nav2_bringup package
    nav2_bringup_dir = FindPackageShare('nav2_bringup')

    # 4. Include the standard localization_launch.py
    localization_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([nav2_bringup_dir, 'launch', 'localization_launch.py'])
        ),
        launch_arguments={
            'map': map_file,
            'params_file': params_file,  
            'use_sim_time': 'false',
            'autostart': 'true',
            'use_lifecycle_mgr': 'true'
        }.items()
    )

    return LaunchDescription([
        localization_cmd
    ])