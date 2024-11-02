import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import Empty
from sensor_msgs.msg import Joy
from vision_msgs.msg import Detection2DArray
from controller_manager_msgs.srv import SwitchController
from tf.transformations import quaternion_from_euler
import numpy as np

KP = 10.0
KD = -0.1
IMAGE_WIDTH = 1400
DETECTION_TIMEOUT = 1.0
SEARCH_YAW_VEL = 0.5
MAX_YAW_VEL = 1.0
CHASE_VEL = 1.0

def main(args=None):
    rclpy.init(args=args)
    node = PupperDemoNode()
    rclpy.spin(node)
    rclpy.shutdown()


class PupperDemoNode(Node):
    def __init__(self):
        super().__init__('pupper_demo')

        # Publishers
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pose_pub = self.create_publisher(Pose, '/cmd_pose', 10)
        self.estop_pub = self.create_publisher(Empty, '/emergency_stop', 10)
        self.estop_reset_pub = self.create_publisher(Empty, '/emergency_stop_reset', 10)

        # Subscribers
        self.joy_sub = self.create_subscription(
            Joy,
            'joy',
            self.joy_callback,
            10)

        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/detections',
            self.detection_callback,
            10)

        # Service clients
        self.switch_controller = self.create_client(
            SwitchController,
            '/controller_manager/switch_controller')

        # State
        self.mode = 'manual'  # manual, tracking, trajectory
        self.estop_button = 7  # Adjust button mapping as needed
        self.estop_reset_button = 6
        self.tracking_button = 5
        self.trajectory_button = 4

        # Visual servoing state
        self.last_detection_time = self.get_clock().now()
        self.last_pos = 0.5
        self.last_time = self.get_clock().now()
        self.last_detection_sign = 1.0
        self.current_bbox_center = 0.5
        self.has_detection = False
        self.timer = self.create_timer(0.1, self.timer_callback)

    def calculate_pd_control(self, pos, target_pos, current_time):
        error = target_pos - pos
        dt = (current_time - self.last_time).nanoseconds / 1e9

        if dt < DETECTION_TIMEOUT and self.last_pos != 0.5:
            derivative = (target_pos - self.last_pos) / dt
        else:
            derivative = 0.0

        self.last_pos = pos
        self.last_time = current_time
        self.last_detection_sign = np.sign(error)

        return KP * error + KD * derivative

    def detection_callback(self, msg):
        self.has_detection = len(msg.detections) > 0

        if self.has_detection:
            self.last_detection_time = self.get_clock().now()
            detection = msg.detections[0]
            self.current_bbox_center = detection.bbox.center.position.x / IMAGE_WIDTH

    def timer_callback(self):
        if self.mode != 'tracking':
            return

        current_time = self.get_clock().now()
        cmd = Twist()

        if not self.has_detection:
            time_since_detection = (current_time - self.last_detection_time).nanoseconds / 1e9
            if time_since_detection > DETECTION_TIMEOUT:
                cmd.angular.z = self.last_detection_sign * SEARCH_YAW_VEL
                self.last_pos = 0.5
        else:
            image_center = 0.5
            yaw_command = self.calculate_pd_control(
                pos=self.current_bbox_center,
                target_pos=image_center,
                current_time=current_time
            )
            yaw_command = np.clip(yaw_command, -MAX_YAW_VEL, MAX_YAW_VEL)

            cmd.angular.z = yaw_command
            cmd.linear.x = CHASE_VEL

        self.vel_pub.publish(cmd)

    def joy_callback(self, msg):
        # E-stop handling
        if msg.buttons[self.estop_button]:
            self.estop_pub.publish(Empty())
            return

        if msg.buttons[self.estop_reset_button]:
            self.estop_reset_pub.publish(Empty())
            return

        # Mode handling
        if msg.buttons[self.tracking_button]:
            self.mode = 'tracking'
            cmd_pose = Pose()
            pitch = -0.52 # Look up during tracking
            q = quaternion_from_euler(0, pitch, 0)
            cmd_pose.orientation.x = q[0]
            cmd_pose.orientation.y = q[1]
            cmd_pose.orientation.z = q[2]
            cmd_pose.orientation.w = q[3]
            self.pose_pub.publish(cmd_pose)

        elif msg.buttons[self.trajectory_button]:
            self.mode = 'trajectory'
            self.execute_trajectory()
        else:
            self.mode = 'manual'

        # Manual control
        if self.mode == 'manual':
            # Velocity command
            cmd_vel = Twist()
            cmd_vel.linear.x = msg.axes[1]  # Left stick Y
            cmd_vel.linear.y = msg.axes[0]  # Left stick X
            cmd_vel.angular.z = msg.axes[2]  # Right stick X
            self.vel_pub.publish(cmd_vel)

            # Pose command (pitch from right stick Y)
            cmd_pose = Pose()
            pitch = msg.axes[3]  # Right stick Y
            q = quaternion_from_euler(0, pitch, 0)
            cmd_pose.orientation.x = q[0]
            cmd_pose.orientation.y = q[1]
            cmd_pose.orientation.z = q[2]
            cmd_pose.orientation.w = q[3]
            self.pose_pub.publish(cmd_pose)

    async def execute_trajectory(self):
        # Switch to trajectory controller
        req = SwitchController.Request()
        req.start_controllers = ['joint_trajectory_controller']
        req.stop_controllers = ['neural_controller']
        req.strictness = 2
        req.start_asap = True
        req.timeout = 0.0

        await self.switch_controller.call_async(req)

        # Execute predefined trajectory here
        # TODO: Add trajectory execution code

        # Switch back to neural controller
        req.start_controllers = ['neural_controller']
        req.stop_controllers = ['joint_trajectory_controller']
        await self.switch_controller.call_async(req)
        self.mode = 'manual'


if __name__ == '__main__':
    main()
