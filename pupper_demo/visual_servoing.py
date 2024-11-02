import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from vision_msgs.msg import Detection2DArray
import numpy as np

KP = 10.0
KD = -0.1
IMAGE_WIDTH = 1400
DETECTION_TIMEOUT = 1.0  # seconds
SEARCH_YAW_VEL = 0.5
MAX_YAW_VEL = 1.0
CHASE_VEL = 1.0

class VisualServoingNode(Node):
    def __init__(self):
        super().__init__('visual_servoing_node')

        self.detection_subscription = self.create_subscription(
            Detection2DArray,
            '/detections',
            self.detection_callback,
            10
        )

        self.command_publisher = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )

        self.timer = self.create_timer(0.1, self.timer_callback)  # 10Hz control loop

        self.last_detection_time = self.get_clock().now()
        self.last_pos = 0.5
        self.last_time = self.get_clock().now()
        self.last_detection_sign = 1.0
        self.current_bbox_center = 0.5
        self.has_detection = False

    def calculate_pd_control(self, pos, target_pos, current_time):
        error = target_pos - pos
        dt = (current_time - self.last_time).nanoseconds / 1e9

        # Calculate derivative term
        if dt < DETECTION_TIMEOUT and self.last_pos != 0.5:
            derivative = (target_pos - self.last_pos) / dt
        else:
            derivative = 0.0

        # Update last values
        self.last_pos = pos
        self.last_time = current_time

        self.last_detection_sign = np.sign(error)

        # Combine P and D terms
        return KP * error + KD * derivative

    def detection_callback(self, msg):
        self.has_detection = len(msg.detections) > 0

        if self.has_detection:
            self.last_detection_time = self.get_clock().now()
            detection = msg.detections[0]
            self.current_bbox_center = detection.bbox.center.position.x / IMAGE_WIDTH

    def timer_callback(self):
        current_time = self.get_clock().now()
        cmd = Twist()

        if not self.has_detection:
            time_since_detection = (current_time - self.last_detection_time).nanoseconds / 1e9
            if time_since_detection > DETECTION_TIMEOUT:
                cmd.angular.z = self.last_detection_sign * SEARCH_YAW_VEL
                self.last_pos = 0.5
        else:
            # Calculate error from image center
            image_center = 0.5

            # Calculate yaw command using PD controller
            yaw_command = self.calculate_pd_control(
                pos=self.current_bbox_center,
                target_pos=image_center,
                current_time=current_time
            )
            yaw_command = np.clip(yaw_command, -MAX_YAW_VEL, MAX_YAW_VEL)

            cmd.angular.z = yaw_command
            cmd.linear.x = CHASE_VEL
            print(yaw_command)

        self.command_publisher.publish(cmd)

def main():
    rclpy.init()
    visual_servoing_node = VisualServoingNode()

    try:
        rclpy.spin(visual_servoing_node)
    except KeyboardInterrupt:
        print("Program terminated by user")
    finally:
        zero_cmd = Twist()
        visual_servoing_node.command_publisher.publish(zero_cmd)

        visual_servoing_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
