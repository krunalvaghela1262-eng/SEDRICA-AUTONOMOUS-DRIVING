import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
import carla
import math

class CarlaMapLinker(Node):
    def __init__(self):
        super().__init__('carla_map_linker')
        
        # 1. SIM TIME: Crucial for syncing with the Bridge
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', True)        

        # 2. CARLA Connection
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.hero_vehicle = None
        
        # 3. Static Broadcaster ONLY
        # We don't use TransformBroadcaster here to avoid hero/odom conflicts
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        
        # 4. Publisher for Ground Truth Data (Optional)
        # Note: We publish to /ground_truth_odom to avoid conflicting with the bridge's /odom
        self.gt_odom_pub = self.create_publisher(Odometry, '/odom', 10)
        
        # Links map -> odom at startup
        self.publish_static_map_to_odom()
        
        # Update Ground Truth message at 50Hz
        self.timer = self.create_timer(0.02, self.update_callback)
        self.get_logger().info("Map Linker Started")

    def publish_static_map_to_odom(self):
        """Creates the bridge between world coordinates and odometry."""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'
        
        # Since we want Ground Truth, odom origin = map origin
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.w = 1.0
        
        self.static_tf_broadcaster.sendTransform(t)

    def update_callback(self):
        if self.get_clock().now().nanoseconds == 0:
            return

        if self.hero_vehicle is None:
            actors = self.world.get_actors().filter('vehicle.*')
            for actor in actors:
                if actor.attributes.get('role_name') == 'hero':
                    self.hero_vehicle = actor
            return

        # Get State for the message
        transform = self.hero_vehicle.get_transform()
        loc = transform.location
        rot = transform.rotation
        now = self.get_clock().now().to_msg()

        # Publish Ground Truth Odometry TOPIC only (No TF)
        # This is useful for evaluation, but won't fight the bridge
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'hero'
        
        odom.pose.pose.position.x = loc.x
        odom.pose.pose.position.y = -loc.y
        odom.pose.pose.position.z = loc.z
        
        # Orientation
        r, p, y = map(math.radians, [rot.roll, -rot.pitch, -rot.yaw])
        cy, sy = math.cos(y*0.5), math.sin(y*0.5)
        cp, sp = math.cos(p*0.5), math.sin(p*0.5)
        cr, sr = math.cos(r*0.5), math.sin(r*0.5)
        
        odom.pose.pose.orientation.w = cr * cp * cy + sr * sp * sy
        odom.pose.pose.orientation.x = sr * cp * cy - cr * sp * sy
        odom.pose.pose.orientation.y = cr * sp * cy + sr * cp * sy
        odom.pose.pose.orientation.z = cr * cp * sy - sr * sp * cy

        self.gt_odom_pub.publish(odom)

def main(args=None):
    rclpy.init(args=args)
    node = CarlaMapLinker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()