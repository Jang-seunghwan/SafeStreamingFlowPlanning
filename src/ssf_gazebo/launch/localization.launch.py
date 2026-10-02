#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([

        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation clock if true'
        ),

        # AMCL
        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[
                {'use_sim_time': use_sim_time},
                {'base_frame_id': 'base_footprint'},
                {'odom_frame_id': 'odom'},
                {'global_frame_id': 'map'},
                {'scan_topic': '/scan'},
                {'tf_broadcast': True}
            ]
        ),

        # Lifecycle Manager (AMCL only)
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_localization',
            output='screen',
            parameters=[
                {'use_sim_time': use_sim_time},
                {'autostart': True},
                {'node_names': ['amcl']}
            ]
        ),
    ])
