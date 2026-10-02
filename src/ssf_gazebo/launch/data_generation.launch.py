#!/usr/bin/env python3
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def _launch_setup(context, *args, **kwargs):
    world = LaunchConfiguration('world').perform(context)
    robot = LaunchConfiguration('robot').perform(context)
    name = LaunchConfiguration('name').perform(context)
    x = LaunchConfiguration('x').perform(context)
    y = LaunchConfiguration('y').perform(context)
    z = LaunchConfiguration('z').perform(context)
    verbose = LaunchConfiguration('verbose').perform(context)
    spawn_delay = LaunchConfiguration('spawn_delay').perform(context)
    headless = LaunchConfiguration('headless').perform(context)

    rollout_count = LaunchConfiguration('rollout_count').perform(context)
    logs_dir = LaunchConfiguration('logs_dir').perform(context)
    generator_start_delay = LaunchConfiguration('generator_start_delay').perform(context)
    max_rollout_time = LaunchConfiguration('max_rollout_time').perform(context)
    inter_rollout_delay = LaunchConfiguration('inter_rollout_delay').perform(context)
    goal_tolerance = LaunchConfiguration('goal_tolerance').perform(context)
    min_goal_distance = LaunchConfiguration('min_goal_distance').perform(context)
    random_seed = LaunchConfiguration('random_seed').perform(context)
    map_yaml = LaunchConfiguration('map').perform(context)

    ssf_share = get_package_share_directory('ssf_gazebo')
    gz_launch_path = os.path.join(ssf_share, 'launch', 'gz_sim.launch.py')
    map_server_launch_path = os.path.join(ssf_share, 'launch', 'map_server.launch.py')
    localization_launch_path = os.path.join(ssf_share, 'launch', 'localization.launch.py')
    navigation_launch_path = os.path.join(ssf_share, 'launch', 'navigation.launch.py')

    if map_yaml:
        map_yaml = os.path.expanduser(map_yaml)
    else:
        map_yaml = os.path.join(ssf_share, 'maps', f'ssf_map_{world}.yaml')

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gz_launch_path),
        launch_arguments={
            'world': world,
            'robot': robot,
            'name': name,
            'x': x,
            'y': y,
            'z': z,
            'verbose': verbose,
            'spawn_delay': spawn_delay,
            'headless': headless,
        }.items(),
    )

    map_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(map_server_launch_path),
        launch_arguments={
            'map': map_yaml,
            'use_sim_time': 'true',
        }.items(),
    )

    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(localization_launch_path),
        launch_arguments={'use_sim_time': 'true'}.items(),
    )

    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(navigation_launch_path),
        launch_arguments={'use_sim_time': 'true'}.items(),
    )

    generator = Node(
        package='ssf_gazebo',
        executable='rollout_data_generator',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'rollout_count': int(rollout_count),
            'logs_dir': logs_dir,
            'max_rollout_time': float(max_rollout_time),
            'inter_rollout_delay': float(inter_rollout_delay),
            'goal_tolerance': float(goal_tolerance),
            'min_goal_distance': float(min_goal_distance),
            'random_seed': int(random_seed),
            'initial_pose_x': float(x),
            'initial_pose_y': float(y),
            'initial_pose_yaw': 0.0,
            # Respawn parameters
            'robot_name': name,
            'world_name': world,
            'spawn_z': float(z),
            'amcl_convergence_delay': 2.0,
        }],
    )

    delayed_generator = TimerAction(
        period=float(generator_start_delay),
        actions=[generator],
    )

    return [sim, map_server, localization, navigation, delayed_generator]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='warehouse'),
        DeclareLaunchArgument('robot', default_value='mecanum'),
        DeclareLaunchArgument('name', default_value='robot'),
        DeclareLaunchArgument('x', default_value='1.0'),
        DeclareLaunchArgument('y', default_value='1.0'),
        DeclareLaunchArgument('z', default_value='0.2'),
        DeclareLaunchArgument('verbose', default_value='4'),
        DeclareLaunchArgument('spawn_delay', default_value='2.0'),
        DeclareLaunchArgument('headless', default_value='false',
                              description='Run gz server-only (no GUI window).'),
        DeclareLaunchArgument('map', default_value=''),

        DeclareLaunchArgument('rollout_count', default_value='10000'),
        DeclareLaunchArgument('logs_dir', default_value='logs',
                              description='Output directory of rollout_*.csv (relative to the cwd)'),
        DeclareLaunchArgument('generator_start_delay', default_value='5.0'),
        DeclareLaunchArgument('max_rollout_time', default_value='150.0'),
        DeclareLaunchArgument('inter_rollout_delay', default_value='2.0'),
        DeclareLaunchArgument('goal_tolerance', default_value='0.1'),
        DeclareLaunchArgument('min_goal_distance', default_value='15.0'),
        DeclareLaunchArgument('random_seed', default_value='42'),
        OpaqueFunction(function=_launch_setup),
    ])
