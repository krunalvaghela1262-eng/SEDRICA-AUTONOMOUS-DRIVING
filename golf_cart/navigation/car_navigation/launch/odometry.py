import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import math

class CarlaOdom(Node):
    def __init__(self):
        super().__init__('carla_odom')
        self.sub = self.create_subscription(
            Odometry,
            '/carla/hero/odometry',
            self.cb,
            10)

        self.pub = self.create_publisher(Odometry, '/odom', 10)
        self.tf = TransformBroadcaster(self)

        self.initialized = False

    def cb(self, msg):
        if not self.initialized:
            self.x0 = msg.pose.pose.position.x
            self.y0 = msg.pose.pose.position.y
            self.initialized = True

        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'hero'

        odom.pose.pose.position.x = msg.pose.pose.position.x - self.x0
        odom.pose.pose.position.y = msg.pose.pose.position.y - self.y0
        odom.pose.pose.orientation = msg.pose.pose.orientation
        odom.twist = msg.twist

        self.pub.publish(odom)

        t = TransformStamped()
        t.header = odom.header
        t.child_frame_id = 'hero'
        t.transform.translation.x = odom.pose.pose.position.x
        t.transform.translation.y = odom.pose.pose.position.y
        t.transform.rotation = odom.pose.pose.orientation
        self.tf.sendTransform(t)
