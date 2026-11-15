import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml

def generate_launch_description():
    rover_dir = get_package_share_directory('car_navigation')

    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    params_file = LaunchConfiguration('params_file')
    log_level = LaunchConfiguration('log_level')

    lifecycle_nodes = [
        # 'amcl',         # Added AMCL here
        'controller_server',
        'planner_server',
        'behavior_server',
        'bt_navigator',
        'velocity_smoother',
        # 'map_server',
        # 'collision_monitor'
    ]

    remappings = [('/tf', 'tf'), ('/tf_static', 'tf_static')]

#     remappings = [
#     # LiDAR
#     ('/scan', '/carla/hero/top_lidar'),
#     ('scan_filtered', '/carla/hero/top_lidar'),

#     # Odometry
#     ('/odom', '/carla/hero/odometry'),
#     ('odometry/filtered', '/carla/hero/odometry'),
# ]


    param_substitutions = {
        'use_sim_time': use_sim_time,
        'autostart': autostart
    }

    configured_params = RewrittenYaml(
        source_file=params_file,
        root_key=LaunchConfiguration('namespace'),
        param_rewrites=param_substitutions,
        convert_types=True
    )

    return LaunchDescription([
        SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '1'),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation clock if true'),

        DeclareLaunchArgument(
            'params_file',
            default_value=os.path.join(rover_dir, 'config', 'nav2_params.yaml'),
            # default_value=os.path.join(rover_dir, 'config', 'gahan_params.yaml'),
            description='Full path to the ROS2 parameters file to use'),

        DeclareLaunchArgument(
            'autostart',
            default_value='true',
            description='Automatically startup the nav2 stack'),

        DeclareLaunchArgument(
            'log_level',
            default_value='info',
            description='log level'),
            
        DeclareLaunchArgument(
            'namespace',
            default_value='',
            description='Top-level namespace'),
        
        # # # Controller Server (Local Planner)
        Node(
            package='nav2_controller',
            executable='controller_server',
            output='screen',
            parameters=[configured_params],
            arguments=['--ros-args', '--log-level', log_level],
            # Remap output to cmd_vel_nav so smoother can catch it
            remappings=remappings + [('cmd_vel', 'cmd_vel_nav')]),

        # Planner Server (Global Planner)
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[configured_params],
            arguments=['--ros-args', '--log-level', log_level],
            remappings=remappings),
            

        # # Behavior Server (Recoveries)
        Node(
            package='nav2_behaviors',
            executable='behavior_server',
            name='behavior_server',
            output='screen',
            parameters=[configured_params],
            arguments=['--ros-args', '--log-level', log_level],
            # Remap output to cmd_vel_nav so smoother can catch it
            remappings=remappings + [('cmd_vel', 'cmd_vel_nav')]),

        # # BT Navigator (Orchestrator)
        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            output='screen',
            parameters=[configured_params],
            arguments=['--ros-args', '--log-level', log_level],
            remappings=remappings),
            
        # # Velocity Smoother 
        Node(
            package='nav2_velocity_smoother',
            executable='velocity_smoother',
            name='velocity_smoother',
            output='screen',
            parameters=[configured_params],
            arguments=['--ros-args', '--log-level', log_level],
            # Reads cmd_vel (remapped to cmd_vel_nav) from controller
            remappings=remappings + [('cmd_vel', 'cmd_vel_nav')]),

        # Node(
        #     package='tf2_ros',
        #     executable='static_transform_publisher',
        #     name='map_to_odom_tf',
        #     arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        # ),

        # Node(
        #     package='nav2_collision_monitor',
        #     executable='collision_monitor',
        #     name='collision_monitor',
        #     output='screen',
        #     parameters=[configured_params],
        #     arguments=['--ros-args', '--log-level', log_level],
        #     remappings=remappings),
    
        

        # Lifecycle Manager
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            arguments=['--ros-args', '--log-level', log_level],
            parameters=[{'autostart': autostart},
                        {'node_names': lifecycle_nodes}]),

        # Node(
        #     package='map_builder',
        #     executable='map_pub',
        #     name='carla_map_to_grid',
        #     output='screen',
        #     parameters=[{'use_sim_time': use_sim_time}]
        # ),

        Node(
            package='car_navigation',
            executable='twist_to_control',
            name='twist_to_control',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}]
        ),
        # Node(
        #     package='tf2_ros',
        #     executable='static_transform_publisher',
        #     name='static_map_to_odom',
        #     arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom']
        # ),
        # Node(
        #     package='nav2_map_server',
        #     executable='map_server',
        #     name='map_server',
        #     output='screen',
        #     parameters=[{'use_sim_time': True}, configured_params]
        # ),
        # --- AMCL Node ---
        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[configured_params],
            arguments=['--ros-args', '--log-level', log_level],
            remappings=remappings
        ),
        # --- PointCloud to LaserScan Converter ---
        Node(
            package='pointcloud_to_laserscan',
            executable='pointcloud_to_laserscan_node',
            name='pointcloud_to_laserscan',
            output='screen',
            parameters=[{
                'target_frame': 'hero',      # Keep it in robot base frame
                'transform_tolerance': 0.01,
                'min_height': 0.1,           # Ignore the ground
                'max_height': 1.0,           # Only see things at "bumper" height
                'angle_min': -3.1415,        # -180 degrees
                'angle_max': 3.1415,         # 180 degrees
                'angle_increment': 0.0087,   # 0.5 degrees per ray
                'scan_time': 0.1,
                'range_min': 0.5,
                'range_max': 50.0,
                'use_inf': True,
                'inf_is_valid': False,
                'use_sim_time': True,
            }],
            remappings=[
                ('cloud_in', '/carla/hero/top_lidar'), # Your Carla input
                ('scan', '/carla/hero/scan')           # The output for AMCL
            ]
        ),

    ])
