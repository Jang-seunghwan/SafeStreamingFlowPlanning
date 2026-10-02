#!/usr/bin/env python3
"""Common utilities for the sim benchmark.

Includes:
  - teleport_robot()    : Gazebo /world/<world>/set_pose via `ign service`
  - publish_initialpose(): re-localize AMCL after teleport
  - clear_costmaps()    : Nav2 global+local costmap clear
  - build_path_msg()    : numpy (H, 4) trajectory → nav_msgs/Path
  - OdomIntegrator      : record odom and integrate driven distance
  - CmdVelLog           : record the /cmd_vel stream (smoothness metric)
  - send_follow_path()  : Nav2 FollowPath action client
"""
from __future__ import annotations
import math
import subprocess
import time
from dataclasses import dataclass, field
from typing import List

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry, Path
from nav2_msgs.srv import ClearEntireCostmap
from nav2_msgs.action import FollowPath


# ---------------------------------------------------------------------------
# Gazebo robot teleport via ign-service
# ---------------------------------------------------------------------------

def teleport_robot(world: str, name: str, x: float, y: float, z: float = 0.05,
                   yaw: float = 0.0, timeout_ms: int = 1500) -> bool:
    """Call ign service /world/<world>/set_pose to move the robot.

    Returns True on success. Uses subprocess to bypass needing an ign-transport
    Python binding; the protobuf request string is the canonical text format.
    """
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)
    req = (
        f'name: "{name}", '
        f'position: {{x: {x}, y: {y}, z: {z}}}, '
        f'orientation: {{x: 0, y: 0, z: {qz}, w: {qw}}}'
    )
    try:
        result = subprocess.run(
            ['ign', 'service',
             '-s', f'/world/{world}/set_pose',
             '--reqtype', 'ignition.msgs.Pose',
             '--reptype', 'ignition.msgs.Boolean',
             '--timeout', str(timeout_ms),
             '--req', req],
            capture_output=True, text=True, timeout=(timeout_ms / 1000.0) + 2.0,
        )
        return ('data: true' in result.stdout) or (result.returncode == 0)
    except subprocess.TimeoutExpired:
        return False


# ---------------------------------------------------------------------------
# AMCL re-localization
# ---------------------------------------------------------------------------

def publish_initialpose(node: Node, x: float, y: float, yaw: float,
                        cov_pos: float = 0.25, cov_yaw: float = 0.07,
                        frame_id: str = 'map',
                        repeat: int = 5) -> None:
    """Publish a PoseWithCovarianceStamped on /initialpose so AMCL resets.

    Repeats publishes a few times because /initialpose is not latched and
    AMCL may miss a single message if discovery isn't complete yet.
    Covariance defaults raised to (0.25 m, 0.07 rad) so AMCL re-samples a
    wider particle cloud — better recovery after a large teleport.
    """
    pub = node.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
    # Wait for AMCL to subscribe.
    deadline = time.monotonic() + 5.0
    while pub.get_subscription_count() < 1 and time.monotonic() < deadline:
        time.sleep(0.1)
    # Build the message once.
    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = frame_id
    msg.pose.pose.position.x = float(x)
    msg.pose.pose.position.y = float(y)
    qz = math.sin(yaw / 2.0); qw = math.cos(yaw / 2.0)
    msg.pose.pose.orientation.z = qz
    msg.pose.pose.orientation.w = qw
    cov = [0.0] * 36
    cov[0] = cov_pos
    cov[7] = cov_pos
    cov[35] = cov_yaw
    msg.pose.covariance = cov
    for _ in range(repeat):
        msg.header.stamp = node.get_clock().now().to_msg()
        pub.publish(msg)
        time.sleep(0.15)


# ---------------------------------------------------------------------------
# Costmap clear
# ---------------------------------------------------------------------------

def clear_costmaps(node: Node, timeout: float = 3.0) -> bool:
    """Call /global_costmap/.../clear_entirely and /local_costmap/.../clear_entirely.

    Service type is nav2_msgs/srv/ClearEntireCostmap (NOT std_srvs/Empty).
    """
    ok = True
    for srv in ['/global_costmap/clear_entirely_global_costmap',
                '/local_costmap/clear_entirely_local_costmap']:
        cli = node.create_client(ClearEntireCostmap, srv)
        if not cli.wait_for_service(timeout_sec=timeout):
            node.get_logger().warn(f'Service not available: {srv}')
            ok = False
            continue
        fut = cli.call_async(ClearEntireCostmap.Request())
        rclpy.spin_until_future_complete(node, fut, timeout_sec=timeout)
        if not fut.done():
            ok = False
    return ok


# ---------------------------------------------------------------------------
# Path message builder
# ---------------------------------------------------------------------------

def build_path_msg(plan: np.ndarray, node: Node, frame_id: str = 'map') -> Path:
    """numpy (H, 2 or 4) trajectory → nav_msgs/Path."""
    msg = Path()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = frame_id
    last_yaw = 0.0
    for i, pt in enumerate(plan):
        ps = PoseStamped()
        ps.header = msg.header
        ps.pose.position.x = float(pt[0])
        ps.pose.position.y = float(pt[1])
        if i + 1 < len(plan):
            dy = float(plan[i + 1, 1] - pt[1])
            dx = float(plan[i + 1, 0] - pt[0])
            if abs(dx) + abs(dy) > 1e-6:
                last_yaw = math.atan2(dy, dx)
        ps.pose.orientation.z = math.sin(last_yaw / 2.0)
        ps.pose.orientation.w = math.cos(last_yaw / 2.0)
        msg.poses.append(ps)
    return msg


# ---------------------------------------------------------------------------
# Odom logger
# ---------------------------------------------------------------------------

@dataclass
class OdomLog:
    times: List[float] = field(default_factory=list)
    x: List[float] = field(default_factory=list)
    y: List[float] = field(default_factory=list)
    yaw: List[float] = field(default_factory=list)
    vx: List[float] = field(default_factory=list)
    vy: List[float] = field(default_factory=list)
    distance_m: float = 0.0

    def add(self, t, x, y, yaw, vx, vy):
        if self.x:
            self.distance_m += math.hypot(x - self.x[-1], y - self.y[-1])
        self.times.append(t)
        self.x.append(x); self.y.append(y); self.yaw.append(yaw)
        self.vx.append(vx); self.vy.append(vy)


def _yaw_from_quat(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class CmdVelLog:
    """Subscribe a Twist topic and record (t, vx, vy) tuples for smoothness metric."""
    def __init__(self, node: Node, topic: str = '/cmd_vel'):
        self.node = node
        self.times: List[float] = []
        self.vx: List[float] = []
        self.vy: List[float] = []
        self.sub = node.create_subscription(Twist, topic, self._cb, 50)

    def _cb(self, msg: Twist):
        t = self.node.get_clock().now().nanoseconds * 1e-9
        self.times.append(t)
        self.vx.append(msg.linear.x)
        self.vy.append(msg.linear.y)

    def reset(self):
        self.times.clear(); self.vx.clear(); self.vy.clear()


class OdomIntegrator:
    """Subscribe /odom and accumulate (t, x, y, yaw, vx_body, vy_body) + travelled distance."""
    def __init__(self, node: Node, topic: str = '/odom'):
        self.node = node
        self.log = OdomLog()
        self._last_xy = None
        self.sub = node.create_subscription(Odometry, topic, self._cb, 50)
        self.have_data = False

    def _cb(self, msg: Odometry):
        self.have_data = True
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = _yaw_from_quat(q.x, q.y, q.z, q.w)
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.log.add(t, x, y, yaw, vx, vy)

    def reset(self):
        self.log = OdomLog()

    @property
    def xy(self):
        if not self.log.x:
            return None
        return (self.log.x[-1], self.log.y[-1])

    @property
    def yaw(self):
        return self.log.yaw[-1] if self.log.yaw else 0.0


# ---------------------------------------------------------------------------
# Nav2 FollowPath action client
# ---------------------------------------------------------------------------

def send_follow_path(node: Node, path_msg: Path, controller_id: str = 'FollowPath',
                     timeout: float = 90.0):
    """Send /follow_path action goal; returns (goal_handle, future_result_or_none)."""
    client = ActionClient(node, FollowPath, 'follow_path')
    if not client.wait_for_server(timeout_sec=5.0):
        node.get_logger().error('follow_path action server not available')
        return None, None
    goal = FollowPath.Goal()
    goal.path = path_msg
    goal.controller_id = controller_id
    send_fut = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, send_fut, timeout_sec=5.0)
    goal_handle = send_fut.result()
    if goal_handle is None or not goal_handle.accepted:
        node.get_logger().warn('follow_path goal rejected')
        return None, None
    result_fut = goal_handle.get_result_async()
    return goal_handle, result_fut


def stop_robot(node: Node, cmd_vel_topic: str = '/cmd_vel'):
    """Publish a zero Twist."""
    pub = node.create_publisher(Twist, cmd_vel_topic, 10)
    time.sleep(0.1)
    pub.publish(Twist())


def dist_xy(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
