#!/usr/bin/env python3
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


class OdomTfBroadcaster(Node):
    def __init__(self):
        super().__init__('odom_tf_broadcaster')
        self._odom_frame_default = self.declare_parameter('odom_frame', 'odom').value
        self._base_frame_default = self.declare_parameter('base_frame', 'base_footprint').value

        self._tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(Odometry, '/odom', self._odom_callback, 50)

    def _odom_callback(self, msg: Odometry) -> None:
        tf_msg = TransformStamped()
        tf_msg.header.stamp = msg.header.stamp
        tf_msg.header.frame_id = msg.header.frame_id or self._odom_frame_default
        tf_msg.child_frame_id = msg.child_frame_id or self._base_frame_default

        tf_msg.transform.translation.x = msg.pose.pose.position.x
        tf_msg.transform.translation.y = msg.pose.pose.position.y
        tf_msg.transform.translation.z = msg.pose.pose.position.z
        tf_msg.transform.rotation = msg.pose.pose.orientation

        self._tf_broadcaster.sendTransform(tf_msg)


def main(args=None):
    rclpy.init(args=args)
    node = OdomTfBroadcaster()
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
