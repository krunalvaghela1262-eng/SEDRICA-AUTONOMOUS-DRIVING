import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from carla_msgs.msg import CarlaEgoVehicleControl

class TwistToControl(Node):
    def __init__(self):
        super().__init__('twist_to_control')
        
        # Subscribe to Nav2 cmd_vel
        self.cmd_sub = self.create_subscription(
            Twist,
            '/cmd_vel_smoothed',
            self.cmd_callback,
            10
        )
        
        # Publish to CARLA vehicle control
        self.control_pub = self.create_publisher(
            CarlaEgoVehicleControl,
            '/carla/hero/vehicle_control_cmd',
            10
        )
        
        self.get_logger().info("Twist to Carla Control Node Started (hero)")

    def cmd_callback(self, msg: Twist):
        control_msg = CarlaEgoVehicleControl()
        
        # Linear velocity (forward/backward)
        # CARLA uses throttle [0, 1] and brake [0, 1] and reverse [bool]
        # But carla_ros_bridge usually supports setting target velocity if we use a different message,
        # OR we can do a simple mapping.
        # However, Nav2 outputs velocity (m/s).
        # ideally we should use a PID controller or similar, but for a simple start, we might Map.
        # BETTER APPROACH: Use the vehicle_control_cmd, which often accepts a "target_velocity" if configured,
        # BUT standard CarlaEgoVehicleControl is just {throttle, steer, brake, etc.}
        # If we want direct control, we map:
        
        # Very simple open-loop mapping (for validation)
        # target_speed = msg.linear.x
        # target_steer = msg.angular.z
        
        # NOTE: A proper implementation would need to know current speed to decide throttle/brake.
        # For now, let's implement a naive proportional mapping or assume the bridge has a PID mode if we were using a different topic.
        # Since we are outputting to /vehicle_control_cmd, we are sending raw controls.
        
        # Naive approach:
        # Forward:
        target_v = msg.linear.x
        steer_z = msg.angular.z
        
        if target_v >= 0:
            control_msg.throttle = min(abs(target_v), 1.0) # Clamping, obviously not physics-calibrated
            control_msg.reverse = False
            control_msg.brake = 0.0
        else:
            control_msg.throttle = min(abs(target_v), 1.0)
            control_msg.reverse = True
            control_msg.brake = 0.0

        # Steering
        # Nav2 angular.z is rad/s. CARLA steer is [-1, 1].
        # We need a scaling factor.
        # Assuming max turn is around 1 rad/s ?
        control_msg.steer = -max(min(steer_z, 1.0), -1.0) # Invert if needed depending on coord system. ROS standard: Left is positive. CARLA: Left is negative.
        # Creating a simple clamp for now.
        
        self.control_pub.publish(control_msg)

def main(args=None):
    rclpy.init(args=args)
    node = TwistToControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
