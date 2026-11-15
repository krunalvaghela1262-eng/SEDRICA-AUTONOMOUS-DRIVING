import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_car_navigation = get_package_share_directory('car_navigation')
    pkg_nav2 = get_package_share_directory('car_navigation') # reusing our own package for the launch file we modified/viewed earlier if needed, but we have a custom launch too.
    # Actually, we rely on the standard navigation.launch.py we saw in this package.

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # 1. Map Builder Node (from map_builder package)
    map_builder_node = Node(
        package='map_builder',
        executable='map_pub',
        name='carla_map_to_grid',
        output='screen'
    )

    # 2. Twist to Control Adapter
    twist_converter_node = Node(
        package='car_navigation',
        executable='twist_to_control',
        name='twist_to_control',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )
    
    # # 3. Lidar to Scan Adapter
    # lidar_converter_node = Node(
    #     package='car_navigation',
    #     executable='simple_lidar_to_scan',
    #     name='simple_lidar_to_scan',
    #     output='screen',
    #     parameters=[{
    #         'use_sim_time': use_sim_time,
    #         'frame_id': 'hero', # LaserScan frame
    #         'min_height': -1.5,
    #         'max_height': 3.0,
    #         'range_max': 50.0
    #     }]
    # )

    # # 4. Nav2
    # # We use our customized params file
    # params_file = os.path.join(pkg_car_navigation, 'config', 'carla_nav2_params.yaml')
    
    # # We include the existing navigation.launch.py from this package
    # # It takes 'params_file' argument.
    # nav2_launch = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(pkg_car_navigation, 'launch', 'navigation.launch.py')
    #     ),
    #     launch_arguments={
    #         'use_sim_time': use_sim_time,
    #         'params_file': params_file,
    #         'autostart': 'true'
    #     }.items()
    # )

    # TF Static Transform if needed (Lidar -> Base)
    # CARLA Bridge usually provides TF. If not, we might need a static publisher.
    # Assuming Bridge provides /tf.

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo/Carla) clock if true'),
        
        map_builder_node, 
        
        twist_converter_node,
        # lidar_converter_node,
        # nav2_launch
    ])
