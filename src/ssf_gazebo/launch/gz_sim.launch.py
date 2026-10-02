#!/usr/bin/env python3
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, TimerAction, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def _launch_setup(context, *args, **kwargs):
    # --- Args resolved at runtime ---
    world = LaunchConfiguration('world').perform(context)   # worlds/<world>.sdf
    robot = LaunchConfiguration('robot').perform(context)   # mecanum
    name = LaunchConfiguration('name').perform(context)

    x = LaunchConfiguration('x').perform(context)
    y = LaunchConfiguration('y').perform(context)
    z = LaunchConfiguration('z').perform(context)

    verbose = LaunchConfiguration('verbose').perform(context)
    spawn_delay = float(LaunchConfiguration('spawn_delay').perform(context))
    headless = LaunchConfiguration('headless').perform(context).lower() in ('1', 'true', 'yes')

    ssf_share = get_package_share_directory('ssf_gazebo')
    world_path = os.path.join(ssf_share, 'worlds', f'{world}.sdf')

    # ros_gz_sim include
    ros_gz_share = get_package_share_directory('ros_gz_sim')
    ros_gz_launch_path = os.path.join(ros_gz_share, 'launch', 'gz_sim.launch.py')

    # On a headless server (no DISPLAY), pass -s to gz so it runs server-only.
    gz_extra = '-s ' if headless else ''
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(ros_gz_launch_path),
        launch_arguments={'gz_args': f'{gz_extra}{world_path} -r -v {verbose}'}.items(),
    )

    # Select xacro
    if robot == 'mecanum':
        xacro_path = os.path.join(ssf_share, 'urdf', 'mecanum_maze_robot.xacro')
    else:
        raise RuntimeError(f"Unknown robot='{robot}'. Use robot:=mecanum")

    # Generate URDF to /tmp
    urdf_out = f'/tmp/{name}.urdf'
    gen_urdf = ExecuteProcess(
        cmd=['bash', '-lc', f'set -e; ros2 run xacro xacro "{xacro_path}" > "{urdf_out}"'],
        output='screen',
    )

    # Spawn robot (URDF string)
    spawn = ExecuteProcess(
        cmd=['bash', '-lc',
            f'ros2 run ros_gz_sim create '
            f'-name "{name}" -x {x} -y {y} -z {z} '
            f'-file "{urdf_out}"'],
        output='screen',
    )

    # Delay spawn so Gazebo world services are ready
    spawn_after = TimerAction(period=spawn_delay, actions=[gen_urdf, spawn])

    # ros_gz_bridge: ROS <-> Gazebo topic bridge
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        output='screen',
        arguments=[
            f'/model/{name}/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '--ros-args',
            '-r',
            f'/model/{name}/cmd_vel:=/cmd_vel',
        ],
    )

    # Publish TF tree from URDF (base_footprint -> base_link -> sensor links)
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': ParameterValue(Command(['xacro ', xacro_path]), value_type=str),
            'use_sim_time': True,
        }],
    )

    # Publish odom -> base_footprint TF from /odom
    odom_tf_broadcaster = Node(
        package='ssf_gazebo',
        executable='odom_tf_broadcaster',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'odom_frame': 'odom',
            'base_frame': 'base_footprint',
        }],
    )

    # Gazebo sensor frame IDs include model-prefixed names; bridge with static TFs.
    lidar_parent = 'lidar_link'
    lidar_child = f'{name}/base_footprint/lidar'
    imu_parent = 'imu_link'
    imu_child = f'{name}/base_footprint/imu'

    lidar_static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        output='screen',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0',
            '--roll', '0', '--pitch', '0', '--yaw', '0',
            '--frame-id', lidar_parent,
            '--child-frame-id', lidar_child,
        ],
    )

    imu_static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        output='screen',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0',
            '--roll', '0', '--pitch', '0', '--yaw', '0',
            '--frame-id', imu_parent,
            '--child-frame-id', imu_child,
        ],
    )

    return [
        gazebo,
        bridge,
        robot_state_publisher,
        odom_tf_broadcaster,
        lidar_static_tf,
        imu_static_tf,
        spawn_after,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='warehouse',
                              description='World name without .sdf (worlds/warehouse.sdf, real_time_factor 1)'),
        DeclareLaunchArgument('robot', default_value='mecanum',
                              description='Robot to spawn: mecanum'),
        DeclareLaunchArgument('name', default_value='robot',
                              description='Model name in Gazebo'),
        DeclareLaunchArgument('x', default_value='1.0'),
        DeclareLaunchArgument('y', default_value='1.0'),
        DeclareLaunchArgument('z', default_value='0.2'),
        DeclareLaunchArgument('verbose', default_value='4'),
        DeclareLaunchArgument('spawn_delay', default_value='2.0',
                              description='Seconds to wait before spawning robot'),
        DeclareLaunchArgument('headless', default_value='false',
                              description='Run gz server-only (no GUI window). Use on a server '
                                          'with no DISPLAY.'),
        OpaqueFunction(function=_launch_setup),
    ])
