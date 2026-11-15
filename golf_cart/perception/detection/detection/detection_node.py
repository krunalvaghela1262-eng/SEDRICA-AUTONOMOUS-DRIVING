import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray 
from std_msgs.msg import Bool
import math

class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')
        
        self.declare_parameter('emergency_distance_m', 6.0)
        self.declare_parameter('min_distance_filter_m', 0.5)
        self.declare_parameter('max_lateral_m', 1.0)
        self.declare_parameter('max_vertical_m', 0.5)

        self.DANGER_THRESHOLD_M = self.get_parameter('emergency_distance_m').value
        self.MIN_DISTANCE_FILTER_M = self.get_parameter('min_distance_filter_m').value
        self.MAX_LATERAL_M = self.get_parameter('max_lateral_m').value
        self.MAX_VERTICAL_M = self.get_parameter('max_vertical_m').value
        
        self.get_logger().info(
            f'Detection Node started. Z-Stop: {self.DANGER_THRESHOLD_M:.2f}m. '
            f'Z-Filter: {self.MIN_DISTANCE_FILTER_M:.2f}m. '
            f'X-Limit: +/-{self.MAX_LATERAL_M:.2f}m. '
            f'Y-Limit: +/-{self.MAX_VERTICAL_M:.2f}m.'
        )

        self.stb_publisher = self.create_publisher(Bool, '/emergency_stop_cmd', 10)

        self.create_subscription(
            Detection3DArray,
            '/oak/nn/spatial_detections',
            self.detection_callback,
            10
        )

    def detection_callback(self, msg: Detection3DArray):
        """Processes the 3D detections and applies Z, X, and Y constraints."""
        
        is_danger = False
        
        for detection in msg.detections:
            try:
                position = detection.results[0].pose.pose.position
                distance_z = position.z
                lateral_x = position.x
                vertical_y = position.y
                
                class_id = detection.results[0].hypothesis.class_id
            except IndexError:
                continue 
            
            is_close_z = self.MIN_DISTANCE_FILTER_M < distance_z < self.DANGER_THRESHOLD_M
            
            is_in_path_x = math.fabs(lateral_x) < self.MAX_LATERAL_M

            is_at_level_y = math.fabs(vertical_y) < self.MAX_VERTICAL_M

            if is_close_z and is_in_path_x and is_at_level_y:
                
                self.get_logger().warn(
                    f'OBSTACLE ({class_id}) in CORRIDOR! Pos: ({lateral_x:.2f}m, {vertical_y:.2f}m, {distance_z:.2f}m). STB ACTIVATED.'
                )
                is_danger = True
                break

        stb_msg = Bool()
        stb_msg.data = is_danger
        self.stb_publisher.publish(stb_msg)

def main(args=None):
    rclpy.init(args=args)
    emergency_node = DetectionNode()
    rclpy.spin(emergency_node)
    emergency_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()