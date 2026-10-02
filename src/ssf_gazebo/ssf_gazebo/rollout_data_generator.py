#!/usr/bin/env python3
from action_msgs.msg import GoalStatus
import csv
import math
import os
import random
import subprocess
import time
from typing import List

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from nav2_msgs.action import NavigateToPose
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node


def _normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    # ZYX yaw from quaternion
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _quaternion_from_yaw(yaw: float) -> tuple:
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (0.0, 0.0, sy, cy)


class RolloutDataGenerator(Node):
    def __init__(self):
        super().__init__('rollout_data_generator')

        self.rollout_count = int(self.declare_parameter('rollout_count', 1000).value)
        self.logs_dir = str(self.declare_parameter('logs_dir', 'logs').value)
        self.control_rate_hz = float(self.declare_parameter('control_rate_hz', 20.0).value)
        self.inter_rollout_delay = float(self.declare_parameter('inter_rollout_delay', 0.5).value)
        self.goal_tolerance = float(self.declare_parameter('goal_tolerance', 0.25).value)
        self.max_rollout_time = float(self.declare_parameter('max_rollout_time', 100.0).value)
        self.min_goal_distance = float(self.declare_parameter('min_goal_distance', 2.0).value)
        self.goal_frame = str(self.declare_parameter('goal_frame', 'map').value)
        self.navigate_action_name = str(self.declare_parameter('navigate_action_name', 'navigate_to_pose').value)
        self.wait_nav_server_timeout = float(self.declare_parameter('wait_nav_server_timeout', 0.1).value)

        self.initial_pose_x = float(self.declare_parameter('initial_pose_x', 1.0).value)
        self.initial_pose_y = float(self.declare_parameter('initial_pose_y', 1.0).value)
        self.initial_pose_yaw = float(self.declare_parameter('initial_pose_yaw', 0.0).value)
        self.initial_pose_publish_count = int(self.declare_parameter("initial_pose_publish_count", 1).value)

        # Sampling bounds (warehouse-like defaults)
        self.x_min = float(self.declare_parameter('x_min', 1.0).value)
        self.x_max = float(self.declare_parameter('x_max', 24.0).value)
        self.y_min = float(self.declare_parameter('y_min', 1.0).value)
        self.y_max = float(self.declare_parameter('y_max', 19.0).value)

        # Safe aisle zones for warehouse (to avoid spawning in shelves)
        # X-axis aisles: before Row1, between Row1-2, between Row2-3, after Row3
        self.safe_x_aisles = [
            (0.5, 3.7),    # Aisle 1: before Row 1 (x=4.0)
            (4.3, 6.7),    # Aisle 2: between Row 1-2
            (7.3, 9.7),    # Aisle 3: between Row 2-3
            (10.3, 24.0),  # Aisle 4: after Row 3 (includes loading area)
        ]
        # Y-axis aisles: south of shelves, between shelf segments, north of shelves
        # Shelves at y=8.0±1.0, y=11.25±1.0, y=17.25±1.0
        self.safe_y_aisles = [
            (0.5, 6.9),    # South corridor (before y=7.0)
            (12.3, 16.2),  # Middle corridor (between y=12.25 and y=16.25)
            (18.3, 19.5),  # North corridor (after y=18.25)
        ]

        seed = int(self.declare_parameter('random_seed', 42).value)
        self._rng = random.Random(seed)

        # Teleport (respawn) parameters
        self.robot_name = str(self.declare_parameter('robot_name', 'robot').value)
        self.world_name = str(self.declare_parameter('world_name', 'warehouse').value)
        self.spawn_z = float(self.declare_parameter('spawn_z', 0.2).value)
        self.amcl_convergence_delay = float(self.declare_parameter("amcl_convergence_delay", 5.0).value)

        self.logs_dir = os.path.abspath(os.path.expanduser(self.logs_dir))
        os.makedirs(self.logs_dir, exist_ok=True)

        self.odom_sub = self.create_subscription(Odometry, '/odom', self._odom_callback, 50)
        self.amcl_sub = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self._amcl_callback, 50)
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.nav_client = ActionClient(self, NavigateToPose, self.navigate_action_name)
        self.timer = self.create_timer(1.0 / self.control_rate_hz, self._control_loop)

        self.have_odom = False
        self.have_amcl = False
        self.initial_pose_remaining = self.initial_pose_publish_count

        # Global state (map frame) - from AMCL
        self.x = 0.0  # Global X position (map frame)
        self.y = 0.0  # Global Y position (map frame)
        self.yaw = 0.0  # Global yaw angle (map frame)

        # Global velocities (map frame) - transformed from body frame
        self.vx = 0.0  # Global X velocity
        self.vy = 0.0  # Global Y velocity
        self.head_rate = 0.0  # Angular velocity (same in all frames)

        # Body frame velocities (from odometry) - for transformation
        self.vx_body = 0.0
        self.vy_body = 0.0

        self.current_rollout = 0
        self.collected_rollouts = 0
        self.active = False
        self.next_rollout_start_time = 0.0
        self.rollout_start_time = 0.0
        self.start_x = 0.0
        self.start_y = 0.0
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.rows: List[List[float]] = []
        self.goal_handle = None
        self.goal_result_future = None
        self.pending_cancel = False

        # Respawn state tracking
        self.respawn_state = 'idle'  # idle, teleporting, initializing, navigating
        self.localization_init_time = 0.0  # Time when initial pose publishing finished

        self.get_logger().info(
            f'Started rollout data generation: rollout_count={self.rollout_count}, logs_dir={self.logs_dir}'
        )

    def _odom_callback(self, msg: Odometry) -> None:
        """Process odometry to get body frame velocities and angular velocity."""
        self.have_odom = True

        # Store body frame velocities (will be transformed to global frame)
        self.vx_body = msg.twist.twist.linear.x
        self.vy_body = msg.twist.twist.linear.y
        self.head_rate = msg.twist.twist.angular.z

        # Transform body frame velocities to global frame using current yaw
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)
        self.vx = cos_yaw * self.vx_body - sin_yaw * self.vy_body
        self.vy = sin_yaw * self.vx_body + cos_yaw * self.vy_body

    def _amcl_callback(self, msg: PoseWithCovarianceStamped) -> None:
        """Process AMCL pose to get global position and orientation (map frame)."""
        self.have_amcl = True

        # Global position (map frame)
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        # Global orientation (map frame)
        q = msg.pose.pose.orientation
        self.yaw = _yaw_from_quaternion(q.x, q.y, q.z, q.w)

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _publish_initial_pose(self) -> None:
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.goal_frame
        msg.pose.pose.position.x = self.initial_pose_x
        msg.pose.pose.position.y = self.initial_pose_y
        qx, qy, qz, qw = _quaternion_from_yaw(self.initial_pose_yaw)
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        msg.pose.covariance = [
            0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0685,
        ]
        self.initial_pose_pub.publish(msg)

    def _publish_initial_pose_at_start(self) -> None:
        """Publish initial pose at sampled start position for AMCL after respawn."""
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.goal_frame

        # Use sampled start position
        msg.pose.pose.position.x = self.start_x
        msg.pose.pose.position.y = self.start_y

        # Calculate yaw to face goal
        spawn_yaw = math.atan2(self.goal_y - self.start_y, self.goal_x - self.start_x)
        qx, qy, qz, qw = _quaternion_from_yaw(spawn_yaw)
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        msg.pose.covariance = [
            0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0685,
        ]
        self.initial_pose_pub.publish(msg)
        self.get_logger().debug(f'Published initial pose at ({self.start_x:.3f}, {self.start_y:.3f})')

    def _teleport_robot(self) -> None:
        """Teleport robot to start position facing goal using Gazebo service."""
        try:
            # Calculate yaw to face goal from start
            spawn_yaw = math.atan2(self.goal_y - self.start_y, self.goal_x - self.start_x)

            # Calculate quaternion from yaw
            qx, qy, qz, qw = _quaternion_from_yaw(spawn_yaw)

            # Use ign service to set entity pose
            # Note: ignition.msgs.Pose has no "pose" wrapper, use position/orientation directly
            pose_str = (
                f'name: "{self.robot_name}", '
                f'position: {{ x: {self.start_x}, y: {self.start_y}, z: {self.spawn_z} }}, '
                f'orientation: {{ x: {qx}, y: {qy}, z: {qz}, w: {qw} }}'
            )

            result = subprocess.run(
                ['ign', 'service', '-s', f'/world/{self.world_name}/set_pose',
                 '--reqtype', 'ignition.msgs.Pose',
                 '--reptype', 'ignition.msgs.Boolean',
                 '--timeout', '2000',
                 '--req', pose_str],
                capture_output=True,
                text=True,
                timeout=5.0
            )

            if result.stderr:
                self.get_logger().warn(f'Teleport stderr: {result.stderr.strip()}')

            if result.returncode == 0:
                self.get_logger().info(
                    f'Teleported robot to ({self.start_x:.3f}, {self.start_y:.3f}) '
                    f'with yaw={spawn_yaw:.3f} rad, facing goal'
                )
                # Teleportation preserves robot entity and plugins, skip sensor waiting
                # and go directly to localization initialization
                self.respawn_state = 'initializing'
                self.initial_pose_remaining = self.initial_pose_publish_count
                # Wait a bit for Gazebo physics to update
                time.sleep(0.2)
            else:
                self.get_logger().error(
                    f'Failed to teleport robot: returncode={result.returncode}, '
                    f'stderr={result.stderr}'
                )
                self._finish_rollout(status='teleport_failed')

        except Exception as e:
            self.get_logger().error(f'Exception during robot teleport: {e}')
            self._finish_rollout(status='teleport_failed')

    def _sample_aisle_position(self) -> (float, float):
        """Sample a random position from safe aisles to avoid obstacles."""
        # Sample x from safe x-aisles (weighted by width)
        x_widths = [(x_max - x_min) for x_min, x_max in self.safe_x_aisles]
        total_x_width = sum(x_widths)
        x_probs = [w / total_x_width for w in x_widths]

        r = self._rng.random()
        cumulative = 0.0
        selected_x_aisle = self.safe_x_aisles[0]
        for i, prob in enumerate(x_probs):
            cumulative += prob
            if r <= cumulative:
                selected_x_aisle = self.safe_x_aisles[i]
                break

        x_min, x_max = selected_x_aisle
        x = self._rng.uniform(x_min, x_max)

        # Sample y from safe y-aisles (weighted by length)
        y_lengths = [(y_max - y_min) for y_min, y_max in self.safe_y_aisles]
        total_y_length = sum(y_lengths)
        y_probs = [l / total_y_length for l in y_lengths]

        r = self._rng.random()
        cumulative = 0.0
        selected_y_aisle = self.safe_y_aisles[0]
        for i, prob in enumerate(y_probs):
            cumulative += prob
            if r <= cumulative:
                selected_y_aisle = self.safe_y_aisles[i]
                break

        y_min, y_max = selected_y_aisle
        y = self._rng.uniform(y_min, y_max)

        return x, y

    def _sample_goal(self, sx: float, sy: float) -> (float, float):
        """Sample goal position from safe aisles with minimum distance constraint."""
        for _ in range(200):
            gx, gy = self._sample_aisle_position()
            if math.hypot(gx - sx, gy - sy) >= self.min_goal_distance:
                return gx, gy
        # Fallback: sample from aisles even if distance constraint not met
        gx, gy = self._sample_aisle_position()
        return gx, gy

    def _status_name(self, status: int) -> str:
        if status == GoalStatus.STATUS_SUCCEEDED:
            return 'succeeded'
        if status == GoalStatus.STATUS_ABORTED:
            return 'aborted'
        if status == GoalStatus.STATUS_CANCELED:
            return 'canceled'
        return f'status_{status}'

    def _goal_response_cb(self, future) -> None:
        if not self.active:
            return

        goal_handle = future.result()
        if not goal_handle.accepted:
            self._finish_rollout(status='goal_rejected')
            return
        self.goal_handle = goal_handle
        self.goal_result_future = goal_handle.get_result_async()
        self.goal_result_future.add_done_callback(self._goal_result_cb)

    def _goal_result_cb(self, future) -> None:
        if not self.active:
            return

        result = future.result()
        status = result.status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self._finish_rollout(status='succeeded')
            return

        # End current rollout and start a new rollout with a new goal.
        self._finish_rollout(status=self._status_name(status))

    def _send_goal(self) -> None:
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.header.frame_id = self.goal_frame
        goal.pose.pose.position.x = self.goal_x
        goal.pose.pose.position.y = self.goal_y
        desired_yaw = math.atan2(self.goal_y - self.start_y, self.goal_x - self.start_x)
        qx, qy, qz, qw = _quaternion_from_yaw(desired_yaw)
        goal.pose.pose.orientation.x = qx
        goal.pose.pose.orientation.y = qy
        goal.pose.pose.orientation.z = qz
        goal.pose.pose.orientation.w = qw

        send_future = self.nav_client.send_goal_async(goal)
        send_future.add_done_callback(self._goal_response_cb)

    def _start_rollout(self) -> None:
        """Initiate a new rollout with robot respawn."""
        self.current_rollout += 1

        # Sample BOTH start and goal positions from safe aisles
        self.start_x, self.start_y = self._sample_aisle_position()
        self.goal_x, self.goal_y = self._sample_goal(self.start_x, self.start_y)

        self.get_logger().info(
            f'Rollout attempt {self.current_rollout} '
            f'(collected {self.collected_rollouts}/{self.rollout_count}): '
            f'start=({self.start_x:.3f}, {self.start_y:.3f}), '
            f'goal=({self.goal_x:.3f}, {self.goal_y:.3f})'
        )

        # Reset state
        self.rows = []
        self.goal_handle = None
        self.goal_result_future = None
        self.pending_cancel = False
        self.active = True

        # Teleport sequence: teleport → init localization → navigate
        self.respawn_state = 'teleporting'
        self._teleport_robot()

    def _save_rollout(self, status: str, rollout_index: int) -> None:
        file_path = os.path.join(self.logs_dir, f'rollout_{rollout_index:04d}.csv')
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                't', 'x', 'y', 'v_x', 'v_y', 'head_angle', 'head_rate',
                'start_x', 'start_y', 'goal_x', 'goal_y',
            ])
            writer.writerows(self.rows)
        self.get_logger().info(
            f'Saved rollout {self.current_rollout:04d} ({status}, {len(self.rows)} rows): {file_path}'
        )

    def _finish_rollout(self, status: str) -> None:
        if not self.active:
            return
        if self.goal_handle is not None and status not in ('succeeded',):
            self.goal_handle.cancel_goal_async()

        self.active = False
        self.respawn_state = 'idle'  # Reset respawn state
        self.next_rollout_start_time = self._now_sec() + self.inter_rollout_delay

        if status == 'timeout':
            self.get_logger().warn(
                f'Discarding rollout attempt {self.current_rollout} due to timeout; '
                f'collected remains {self.collected_rollouts}/{self.rollout_count}'
            )
        elif status in ('aborted', 'canceled', 'goal_rejected'):
            self.get_logger().warn(
                f'Discarding rollout attempt {self.current_rollout} due to {status}; '
                f'collected remains {self.collected_rollouts}/{self.rollout_count}'
            )
        else:
            self.collected_rollouts += 1
            self._save_rollout(status=status, rollout_index=self.collected_rollouts)

        self.goal_handle = None
        self.goal_result_future = None
        self.pending_cancel = False

    def _control_loop(self) -> None:
        """Main control loop with extended state machine for respawn."""

        # Initial system startup: wait for odom only (AMCL will start after initial pose)
        if not self.have_odom and self.respawn_state == 'idle':
            return

        # Wait for Nav2 server availability
        if not self.nav_client.server_is_ready():
            self.nav_client.wait_for_server(timeout_sec=self.wait_nav_server_timeout)
            return

        # Initial localization setup (only once at startup, not during respawn)
        if self.initial_pose_remaining > 0 and not self.active:
            self._publish_initial_pose()
            self.initial_pose_remaining -= 1
            return

        now = self._now_sec()

        # Check if all rollouts completed
        if self.collected_rollouts >= self.rollout_count and not self.active:
            self.get_logger().info('All rollouts completed. Shutting down node.')
            rclpy.shutdown()
            return

        # Handle respawn state machine
        if self.active:
            if self.respawn_state == 'teleporting':
                # Waiting for teleport subprocess to complete
                # Teleportation happens in _teleport_robot() and transitions to 'initializing'
                return

            elif self.respawn_state == 'initializing':
                # Publish initial pose for AMCL convergence
                if self.initial_pose_remaining > 0:
                    self._publish_initial_pose_at_start()
                    self.initial_pose_remaining -= 1
                    if self.initial_pose_remaining == 0:
                        # Mark time when initial pose publishing finished
                        self.localization_init_time = now
                        self.get_logger().info(f'Initial pose published, waiting {self.amcl_convergence_delay}s for AMCL convergence')
                    return

                # Wait for AMCL to converge before starting navigation
                elapsed_since_init = now - self.localization_init_time
                if elapsed_since_init < self.amcl_convergence_delay:
                    return

                # AMCL should be converged, start navigation
                self.respawn_state = 'navigating'
                self.rollout_start_time = now
                self.rows = [[
                    0.0, self.x, self.y, self.vx, self.vy, self.yaw, self.head_rate,
                    self.start_x, self.start_y, self.goal_x, self.goal_y,
                ]]
                # Verify robot position after teleport
                odom_dist = math.hypot(self.x - self.start_x, self.y - self.start_y)
                self.get_logger().info(
                    f'AMCL converged, starting navigation from odom=({self.x:.3f}, {self.y:.3f}) '
                    f'to goal=({self.goal_x:.3f}, {self.goal_y:.3f}), '
                    f'expected_start=({self.start_x:.3f}, {self.start_y:.3f}), '
                    f'odom_error={odom_dist:.3f}m'
                )
                if odom_dist > 1.0:
                    self.get_logger().error(
                        f'Teleport verification FAILED! Odometry position '
                        f'({self.x:.3f}, {self.y:.3f}) is {odom_dist:.3f}m away from '
                        f'expected start position ({self.start_x:.3f}, {self.start_y:.3f})'
                    )
                self._send_goal()
                return

        # Start new rollout if idle
        if not self.active:
            if now < self.next_rollout_start_time:
                return
            self._start_rollout()
            return

        # Active rollout: collect data and monitor progress
        if self.respawn_state == 'navigating':
            t = now - self.rollout_start_time
            dist = math.hypot(self.goal_x - self.x, self.goal_y - self.y)

            # Log one row per control tick
            self.rows.append([
                t, self.x, self.y, self.vx, self.vy, self.yaw, self.head_rate,
                self.start_x, self.start_y, self.goal_x, self.goal_y,
            ])

            if dist <= self.goal_tolerance:
                self._finish_rollout(status='goal_reached_by_distance')
                return

            if t >= self.max_rollout_time:
                if self.goal_handle is not None and not self.pending_cancel:
                    self.pending_cancel = True
                    self.goal_handle.cancel_goal_async()
                self._finish_rollout(status='timeout')


def main(args=None):
    rclpy.init(args=args)
    node = RolloutDataGenerator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
