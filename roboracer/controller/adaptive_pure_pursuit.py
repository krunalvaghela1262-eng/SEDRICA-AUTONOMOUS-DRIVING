import numpy as np
import rclpy
import rclpy.logging
from rclpy.node import Node
# from geometry_msgs.msg import Point, Twist, Pose
# from sklearn.preprocessing import MinMaxScaler
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from math import atan2, sqrt, pow, sin, exp
import pandas as pd
import time


class AdaptivePurePursuit(Node):
    def __init__(self):
        super().__init__('adaptive_pure_pursuit')

        self.pos_sub = self.create_subscription(
            Odometry, '/odom', self.pos_callback, 3)
        self.thr_pub = self.create_publisher(
            Float32, '/autodrive/f1tenth_1/throttle_command', 3)
        self.str_pub = self.create_publisher(
            Float32, '/autodrive/f1tenth_1/steering_command', 3)
        self.goal_pub = self.create_publisher(Marker, '/goal', 3)
        self.race_pub = self.create_publisher(MarkerArray, '/raceline', 3)
        timer = 1.0
        self.timer = self.create_timer(timer, self.publish_control_commands)
        self.max_speed = 0.07
        self.min_speed = 0.01
        self.max_lookahead = 2.0
        self.min_lookahead = 1.0
        self.wheelbase = 0.33
        self.current_quaternion = [0.0, 0.0, 0.0, 1.0]
        self.lookahead_distance = self.min_lookahead
        self.beta = 0.5
        self.path = np.array([])
        self.previous_position = None
        self.previous_deviation = 0
        self.total_area = 0
        self.area_window = []
        self.window_size = 10
        self.position = None
        self.orientation = None
        self.control_velocity = 0.0015
        self.heading_angle = 0.01

        self.load_raceline_csv(
            '/home/sedrica/docker_ki/auto_api/src/autodrive_devkit/track/gl_track.csv')

    def load_raceline_csv(self, filename):
        self.path = pd.read_csv(filename)
        self.path = np.array([self.path]).reshape(-1, 2)

        for i in range(len(self.path)):
            self.path[i, 1] += 0.5
            self.path[i, 0] -= 2.8

        rotation_matrix = np.array([[0, 1], [-1, 0]])
        self.path = np.dot(self.path, rotation_matrix.T)

    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def update_lookahead(self, speed):
        normalized_speed = (speed - self.min_speed) / \
            (self.max_speed - self.min_speed)
        sigmoid_value = self.sigmoid(normalized_speed * 10 - 5)

        if speed < self.min_speed:
            self.lookahead_distance = self.min_lookahead
        else:
            scaled_lookahead = self.min_lookahead + sigmoid_value * \
                (self.max_lookahead - self.min_lookahead)
            self.lookahead_distance = min(self.max_lookahead, scaled_lookahead)

        print("lookahead_distance", self.lookahead_distance)

    def pos_callback(self, msg):
        rclpy.logging.get_logger('rclpy').info("pos_callback")
        self.position = msg.pose.pose.position
        print("position", self.position)
        self.orientation = msg.pose.pose.orientation

        self.current_quaternion = [
            self.orientation.x, self.orientation.y, self.orientation.z, self.orientation.w]
        self.yaw = self.quaternion_to_yaw(self.current_quaternion)

        current_speed = msg.twist.twist.linear.x

        self.update_lookahead(current_speed)

        closest_point, goal_point = self.get_lookahead_point(self.position)
        print("closest_point", closest_point)
        print("goal_point", goal_point)

        if goal_point is not None:
            alpha = self.calculate_alpha(self.position, goal_point, self.yaw)
            self.heading_angle = self.calculate_heading_angle(alpha)

            area = self.calculate_deviation(self.position, closest_point)

            max_velocity_pp = self.calculate_max_velocity_pure_pursuit(
                self.calculate_curvature(alpha))
            # print("max_velocity_pp", max_velocity_pp)
            min_deviation_pp = self.calculate_min_deviation_pure_pursuit(area)
            # print("min_deviation_pp", min_deviation_pp)

            self.control_velocity = self.convex_combination(
                max_velocity_pp, min_deviation_pp, current_speed, area)

            # print("control_velocity", self.control_velocity)
            self.publish_control_commands()

        # else:
        #     self.control_velocity = 0.7
        #     self.heading_angle = 0.1
        #     self.publish_control_commands()

    def quaternion_to_yaw(self, quaternion):
        qx, qy, qz, qw = quaternion
        siny_cosp = 2*(qw * qz + qx * qy)
        cosy_cosp = 1 - 2*(qy * qy + qz * qz)
        yaw = atan2(siny_cosp, cosy_cosp)

        return yaw

    def get_lookahead_point(self, position):
        rclpy.logging.get_logger('rclpy').info("get_lookahead_point")
        min_dist = float('inf')
        closest_point = None
        goal_point = None
        # print("position", position)
        for point in self.path:
            dist = sqrt(pow(point[0] - position.x, 2) +
                        pow(point[1] - position.y, 2))
            # print("current_dist ", dist)
            if dist < min_dist:
                min_dist = dist
                closest_point = point
        closest_point_index = np.where(self.path == closest_point)

        for point in self.path:
            dist = sqrt(pow(point[0] - position.x, 2) +
                        pow(point[1] - position.y, 2))
            point_index = np.where(self.path == point)
            if dist > self.lookahead_distance and point_index[0][0] > closest_point_index[0][0] + 2 and point_index[0][0] < closest_point_index[0][0] + 6:
                goal_point = point

                marker = Marker()
                marker.header.frame_id = 'world'
                marker.header.stamp = self.get_clock().now().to_msg()

                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = point[0]
                marker.pose.position.y = point[1]
                marker.pose.position.z = 0.0

                marker.scale.x = 0.1
                marker.scale.y = 0.1
                marker.scale.z = 0.1

                marker.color.a = 1.0
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0

                markerarray = MarkerArray()
                for i in range(len(self.path)):
                    marker1 = Marker()
                    marker1.header.frame_id = 'world'
                    marker1.header.stamp = self.get_clock().now().to_msg()

                    marker1.type = Marker.SPHERE
                    marker1.action = Marker.ADD
                    marker1.id = i
                    marker1.pose.position.x = self.path[i][0]
                    marker1.pose.position.y = self.path[i][1]
                    marker1.pose.position.z = 0.0

                    marker1.scale.x = 0.1
                    marker1.scale.y = 0.1
                    marker1.scale.z = 0.1

                    marker1.color.a = 1.0
                    marker1.color.r = 0.0
                    marker1.color.g = 1.0
                    marker1.color.b = 0.0
                    markerarray.markers.append(marker1)

                self.goal_pub.publish(marker)
                self.race_pub.publish(markerarray)

                # print("goal", goal_point)
                break

        return closest_point, goal_point

    def calculate_alpha(self, position, goal_point, yaw):
        dy = goal_point[1] - position.y
        dx = goal_point[0] - position.x
        local_x = dx*np.cos(-yaw) - dy*np.sin(-yaw)
        local_y = dx*np.sin(-yaw) + dy*np.cos(-yaw)
        alpha = atan2(local_y, local_x)
        return alpha

    def calculate_heading_angle(self, alpha):
        heading_angle = atan2(2 * self.wheelbase *
                              sin(alpha), self.lookahead_distance)
        return heading_angle

    def calculate_curvature(self, alpha):
        curvature = 2 * sin(alpha) / self.lookahead_distance
        return curvature

    def calculate_deviation(self, position, closest_point):
        deviation = sqrt(
            pow(closest_point[0] - position.x, 2) + pow(closest_point[1] - position.y, 2))

        if self.previous_position is not None:
            distance_traveled = sqrt(pow(position.x - self.previous_position.x, 2) +
                                     pow(position.y - self.previous_position.y, 2))
            area_increment = (
                deviation + self.previous_deviation) / 2 * distance_traveled

            self.area_window.append(area_increment)
            if len(self.area_window) > self.window_size:
                self.area_window.pop(0)

            self.total_area = sum(self.area_window)

        self.previous_position = position
        self.previous_deviation = deviation

        return self.total_area

    def calculate_max_velocity_pure_pursuit(self, curvature):
        max_velocity = sqrt(
            1 / abs(curvature)) if curvature != 0 else self.max_speed
        return min(self.max_speed, max_velocity)

    def calculate_min_deviation_pure_pursuit(self, area):
        if area > 0:
            min_deviation_velocity = self.max_speed / (1 + area)
        else:
            min_deviation_velocity = self.max_speed
        return min_deviation_velocity

    def convex_combination(self, max_velocity_pp, min_deviation_pp, current_speed, area):
        self.beta = self.adjust_beta(current_speed, area)

        control_velocity = self.beta * max_velocity_pp + \
            (1 - self.beta) * min_deviation_pp

        return control_velocity

    def adjust_beta(self, current_speed, area):
        if area < 1.0:
            return min(1.0, self.beta + 0.25)
        elif current_speed < self.max_speed * 0.3:
            return max(0.0, self.beta - 0.25)
        return self.beta

    def publish_control_commands(self):
        float1 = Float32()
        float2 = Float32()

        if self.control_velocity is None or self.heading_angle is None:
            return

        # min_velocity = 1.0
        # max_velocity = 7.0
        # min_angle = -np.pi
        # max_angle = np.pi

        # normalized_velocity = (self.control_velocity -
        #                        min_velocity) / (max_velocity - min_velocity)
        # normalized_angle = (self.heading_angle - min_angle) / \
        #     (max_angle - min_angle)

        # normalized_velocity = np.clip(normalized_velocity, 0, 1)
        # normalized_angle = np.clip(normalized_angle, 0, 1)

        float1.data = self.control_velocity
        float2.data = self.heading_angle * 2.1

        print("vel:", self.control_velocity,
              "angle:", np.rad2deg(self.heading_angle), "rad:", self.heading_angle)

        self.thr_pub.publish(float1)
        self.str_pub.publish(float2)

        time.sleep(0.2)


def main(args=None):
    rclpy.init(args=args)

    adaptive_pure_pursuit = AdaptivePurePursuit()
    rate = adaptive_pure_pursuit.create_rate(0.1)

    try:
        rclpy.spin(adaptive_pure_pursuit)
        rate.sleep()
    except KeyboardInterrupt:
        pass

    adaptive_pure_pursuit.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
