import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, LaserScan
from sensor_msgs_py import point_cloud2
import math
import numpy as np

class SimpleLidarToScan(Node):
    def __init__(self):
        super().__init__('simple_lidar_to_scan')
        
        self.declare_parameter('min_height', -2.0)
        self.declare_parameter('max_height', 20.0)
        self.declare_parameter('angle_min', -3.14159)
        self.declare_parameter('angle_max', 3.14159)
        self.declare_parameter('range_min', 0.1)
        self.declare_parameter('range_max', 30.0) # CARLA lidar often 10-100m
        self.declare_parameter('scan_time', 0.05) # 20 Hz
        self.declare_parameter('angle_increment', 0.0087) # ~0.5 degree
        self.declare_parameter('frame_id', 'ego_vehicle/lidar') # Ideally output frame
        
        # Subscribe to CARLA Lidar
        self.pcl_sub = self.create_subscription(
            PointCloud2,
            '/carla/hero/semantic_lidar',
            self.pcl_callback,
            10
        )
        
        # Publish LaserScan
        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)
        self.get_logger().info("Simple Lidar to Scan Node Started")

    def pcl_callback(self, msg):
        # Read parameters
        min_h = self.get_parameter('min_height').value
        max_h = self.get_parameter('max_height').value
        angle_min = self.get_parameter('angle_min').value
        angle_max = self.get_parameter('angle_max').value
        range_min = self.get_parameter('range_min').value
        range_max = self.get_parameter('range_max').value
        angle_inc = self.get_parameter('angle_increment').value
        
        scan_msg = LaserScan()
        scan_msg.header = msg.header
        scan_msg.angle_min = angle_min
        scan_msg.angle_max = angle_max
        scan_msg.angle_increment = angle_inc
        scan_msg.time_increment = 0.0
        scan_msg.scan_time = self.get_parameter('scan_time').value
        scan_msg.range_min = range_min
        scan_msg.range_max = range_max
        
        # Calculate size of ranges array
        num_readings = int((angle_max - angle_min) / angle_inc)
        scan_msg.ranges = [float('inf')] * num_readings
        
        # Process points
        # Generator for PointCloud2
        points = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        
        for p in points:
            x, y, z = p
            
            # Simple Pass-Through Filter on Z
            if z < min_h or z > max_h:
                continue
            
            # Cartesian to Polar
            range_val = math.sqrt(x*x + y*y)
            angle = math.atan2(y, x)
            
            if range_val < range_min or range_val > range_max:
                continue
                
            if angle < angle_min or angle > angle_max:
                continue
                
            # Map angle to index
            index = int((angle - angle_min) / angle_inc)
            
            if 0 <= index < num_readings:
                # We want the CLOSEST point in this angular bucket for a conservative obstacle map
                if range_val < scan_msg.ranges[index]:
                    scan_msg.ranges[index] = range_val
                    
        self.scan_pub.publish(scan_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SimpleLidarToScan()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
