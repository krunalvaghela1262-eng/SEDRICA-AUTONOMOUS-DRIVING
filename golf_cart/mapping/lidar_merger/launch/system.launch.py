import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():

    # 1. Include YOUR existing launch file (Loads drivers with your IPs)
    # Assumes your file is at: src/ouster-ros/ouster-ros/launch/launch.py
    # Which installs to: share/ouster_ros/launch/launch.py
    ouster_pkg = get_package_share_directory('ouster_ros')
    existing_drivers_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ouster_pkg, 'launch', 'launch.py'))
    )

    # 2. Publish Transforms (Based on your manual_calib results)
    # Structure: x y z yaw pitch roll parent_frame child_frame

    # Front -> Top (From fronttop.txt: x=-0.78 y=-0.66 z=-1.25 yaw=0.698)
    tf_front = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_front',
        arguments=['-0.78668', '-0.66314', '-1.25', '0.698', '0', '0', 
                   'os_top_sensor', 'os_front_sensor']
    )

    # Back -> Top (From backtop.txt: x=2.34 y=0.77 z=-1.25 yaw=-2.705)
    tf_back = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_back',
        arguments=['2.34027', '0.778379', '-1.25', '-2.705', '0', '0', 
                   'os_top_sensor', 'os_back_sensor']
    )

    # 3. Run the Merger Node
    merger_node = Node(
        package='lidar_merger',
        executable='merger_node',
        name='merger_node',
        parameters=[{'target_frame': 'os_top_sensor'}]
    )

    # 4. Visualization (Optional)
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz_merged',
        output='screen'
    )

    return LaunchDescription([
        existing_drivers_launch,
        tf_front,
        tf_back,
        TimerAction(period=5.0, actions=[merger_node]), # Wait 5s for drivers
        TimerAction(period=8.0, actions=[rviz_node])
    ])