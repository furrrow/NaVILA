from __future__ import annotations
import time
import argparse
from argparse import Namespace
import cv2
from cv_bridge import CvBridge
import numpy as np
import os
import yaml
from PIL import Image as PILImage
from queue import Queue, Full, Empty
# ROS2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Bool, Float32MultiArray, Empty
from nav_msgs.msg import Path, Odometry
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import tf2_ros
from geometry_msgs.msg import Vector3Stamped, PoseStamped
from scipy.spatial.transform import Rotation as R
from custom_utils.io_utils import load_calibration, overlay_path
import matplotlib
matplotlib.use("Agg")

from NaVILA_inference import NavilaPolicy, navila_command_to_waypoints


class NaVILANode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__('Navila_node')
        self.obs_img = None
        self.current_yaw = None
        # CONSTANTS
        current_dir = os.path.dirname(os.path.abspath(__file__))
        print(current_dir)
        parent_dir = "/home/jim/Projects/NaVILA"
        # parent_dir = "/home/gamma-nav/Documents/Projects/git_repos/NaVILA"
        # parent_dir = "/workspace/NaVILA"
        MODEL_PATH = f"{current_dir}/checkpoints/navila-llama3-8b-8f"
        DEPLOY_CONFIG_PATH = f"{current_dir}/config/robot.yaml"
        CAMERA_MATRIX_DIR = f"{current_dir}/cam_matrix.json"
        with open(DEPLOY_CONFIG_PATH, "r") as f:
            deploy_config = yaml.safe_load(f)
        self.rate = deploy_config["frame_rate"]

        robot_config = deploy_config[args.robot]
        print(f"using robot config for: {args.robot}")
        self.instruction = args.instruction
        print(f"using instruction: {args.instruction}")
        self.waypoint_scale_factor = int(args.scale)
        print(f"scaling waypoints by: {self.waypoint_scale_factor}")
        self.max_v = robot_config["max_v"]
        self.max_w = robot_config["max_w"]
        self.original_img_size = (deploy_config["img_w"], deploy_config["img_h"])  # (1280, 720)
        self.shrink_img_size = (deploy_config["shrink_w"], deploy_config["shrink_h"])  # (640, 480)
        self.detection_queue = []
        self.detection_queue_len = 20
        self.robot_velocity_base = np.zeros(3, dtype=np.float64)
        self.robot_angular_velocity_base = np.zeros(3, dtype=np.float64)
        self.dt = 1 / self.rate
        self.reached_goal = False
        self.path_frame_id = "base_link"
        self._started_sent = False
        self.show_time_performance = False
        self.visualize = False

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self,
        )
        self.inference_count = 0
        self.inference_start_time = time.perf_counter()

        # ROS Topics
        IMAGE_TOPIC = robot_config['image_topic']
        ODOM_TOPIC = robot_config['odom_topic']
        self.compressed_img_topic = True if "compressed" in IMAGE_TOPIC else False
        print(f"IMAGE_TOPIC: {IMAGE_TOPIC} compressed_img_topic: {self.compressed_img_topic}")
        POLICY_PATH_TOPIC = robot_config['policy_path_topic']
        WAYPOINT_TOPIC = robot_config['waypoint_topic']
        SAMPLED_ACTIONS_TOPIC = robot_config['sampled_actions_topic']
        REACHED_GOAL_TOPIC = robot_config['reached_goal_topic']
        OVERLAY_TOPIC = robot_config['overlay_topic']

        self.cam_matrix, self.dist_coeffs, self.T_base_from_cam = load_calibration(CAMERA_MATRIX_DIR)
        self.T_cam_from_base = np.linalg.inv(self.T_base_from_cam)

        # models
        self.policy = NavilaPolicy(MODEL_PATH)
        print("loading model: ", MODEL_PATH)

        # ROS 2 Topics
        msg_type = CompressedImage if self.compressed_img_topic else Image
        self.image_sub = self.create_subscription(
            msg_type, IMAGE_TOPIC, self.img_callback_obs,
            qos_profile=QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                                   history=QoSHistoryPolicy.KEEP_LAST,
                                   depth=10))
        # self.odom_sub = self.create_subscription(
        #     Odometry, ODOM_TOPIC, self.odom_callback_obs,
        #     qos_profile=QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
        #                            history=QoSHistoryPolicy.KEEP_LAST,
        #                            depth=10))
        self.waypoint_pub = self.create_publisher(
            Float32MultiArray, WAYPOINT_TOPIC,
            qos_profile=QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                                   history=QoSHistoryPolicy.KEEP_LAST,
                                   depth=10))
        self.sampled_actions_pub = self.create_publisher(
            Float32MultiArray, SAMPLED_ACTIONS_TOPIC,
            qos_profile=QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                   history=QoSHistoryPolicy.KEEP_LAST,
                                   depth=10))
        self.trajectory_visual_pub = self.create_publisher(
            Image, OVERLAY_TOPIC, qos_profile=QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                                                         history=QoSHistoryPolicy.KEEP_LAST,
                                                         depth=10))
        # self.goal_pub = self.create_publisher(Bool, REACHED_GOAL_TOPIC, 1)
        self.pub_started = self.create_publisher(Empty, "/started", 10)
        self.pub_path = self.create_publisher(Path, POLICY_PATH_TOPIC,
                                              qos_profile=QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                                                                     history=QoSHistoryPolicy.KEEP_LAST,
                                                                     depth=10))
        self.timer = self.create_timer(1.0 / self.rate, lambda: self.run_inference_loop())
        print("Waiting for image observations...")

        self.br = CvBridge()
        # Publish /started once, when we actually start inferencing
        if not self._started_sent:
            self._started_sent = True
            self._have_cur_img = False
            self._have_cur_pose = False
            self.pub_started.publish(Empty())
            self.get_logger().info("Published /started (once).")

    def img_callback_obs(self, msg: Image):
        # self.get_logger().info("Reached Image callback!")
        if self.compressed_img_topic:
            self.obs_img = self.br.compressed_imgmsg_to_cv2(msg)
            self.obs_img = cv2.cvtColor(self.obs_img, cv2.COLOR_BGR2RGB)
        else:
            self.obs_img = self.br.imgmsg_to_cv2(msg)
        # Original camera timestamp
        self.obs_img_timestamp = msg.header.stamp
        self.obs_img = PILImage.fromarray(self.obs_img)
        if self.obs_img.size != self.shrink_img_size:
            # print(f"resizing image from {self.obs_img.size} to {self.shrink_img_size}")
            self.obs_img = self.obs_img.resize(self.shrink_img_size)

    # def odom_callback_obs(self, msg: Odometry):
    #     # self.get_logger().info("Reached Odom callback!")
    #     p = msg.pose.pose.position
    #     q = msg.pose.pose.orientation
    #     yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2]
    #     self.current_pos = np.array([p.x, p.y])
    #     self.current_yaw = yaw
    #     self._have_cur_pose = True
    #     self.robot_velocity_base[:] = [
    #         msg.twist.twist.linear.x,
    #         msg.twist.twist.linear.y,
    #         msg.twist.twist.linear.z,
    #     ]
    #
    #     self.robot_angular_velocity_base[:] = [
    #         msg.twist.twist.angular.x,
    #         msg.twist.twist.angular.y,
    #         msg.twist.twist.angular.z,
    #     ]
    #
    # def get_linear_velocity(self):
    #     return np.array([
    #         self.robot_velocity_base[0],
    #         self.robot_velocity_base[1],
    #         self.robot_velocity_base[2],
    #     ])

    import numpy as np
    from nav_msgs.msg import Path
    from geometry_msgs.msg import PoseStamped

    def _to_path_msg(self, path: np.ndarray) -> Path:
        """
        Convert path to nav_msgs/Path.
        Each waypoint can be:
            [x, y] or [x, y, hx, hy]
        where (hx, hy) is a unit heading vector.
        """
        assert path.ndim == 2
        assert path.shape[1] in (2, 4), \
            "path must have shape (N, 2) or (N, 4)"
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.path_frame_id

        for waypoint in path:
            if len(waypoint) == 2:
                x, y = waypoint
                hx, hy = 1.0, 0.0
            else:
                x, y, hx, hy = waypoint

            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = 0.0

            # heading vector -> yaw
            yaw = np.arctan2(hy, hx)
            # yaw -> quaternion
            ps.pose.orientation.x = 0.0
            ps.pose.orientation.y = 0.0
            ps.pose.orientation.z = float(np.sin(yaw / 2.0))
            ps.pose.orientation.w = float(np.cos(yaw / 2.0))
            msg.poses.append(ps)

        return msg

    def run_inference_loop(self):
        chosen_waypoint = np.zeros(4)
        if self.obs_img is not None:
            # if self.show_time_performance:
            #     t1 = time.perf_counter()
            #     self.get_logger().info(f" === > depth_model inference took {(t1 - t0) * 1000:.1f} ms")

            self.policy.add_frame(self.obs_img)
            output = self.policy.predict(self.instruction)
            self.get_logger().info(f"output: {output}")
            # note, we produce n=waypoint_scale_factor+1 waypoints from navila
            # we also scale produced waypoint by the same waypoint_scale_factor
            # thus the second waypoint is the same command produced by navila prior to scaling
            waypoints = navila_command_to_waypoints(output, num_steps=self.waypoint_scale_factor + 1)
            waypoints *= self.waypoint_scale_factor
            if np.array_equal(waypoints, [[0, 0]]):
                self.get_logger().warn("NaVILA commands stop.")
                path_xy = waypoints
            else:
                if len(waypoints[0]) == 2:
                    path_xy = waypoints[:, :2]
                    self.pub_path.publish(self._to_path_msg(path_xy[1:]))
                    # print(len(path_xy[1:]))
                else: # pure rotation with 4d waypoints
                    path_xy = None
                    self.pub_path.publish(self._to_path_msg(waypoints[1:]))
                    # print(len(waypoints[1:]))
                self.waypoint_idx = 1
                chosen_waypoint = waypoints[self.waypoint_idx]
                self.get_logger().info(f"publishing path # {self.waypoint_idx} of chosen_waypoint: {chosen_waypoint}")

            t4 = time.perf_counter()
            # visualization code
            if self.visualize:
                if path_xy is None:
                    overlay_img = np.array(self.obs_img.resize(self.original_img_size))
                else:
                    overlay_img = overlay_path(trajectories=path_xy,
                                               img=np.array(self.obs_img.resize(self.original_img_size)),
                                               cam_matrix=self.cam_matrix,
                                               T_cam_from_base=self.T_cam_from_base, )
                out_msg = self.br.cv2_to_imgmsg(np.array(overlay_img), encoding="rgb8")
                self.trajectory_visual_pub.publish(out_msg)
                if self.show_time_performance:
                    t5 = time.perf_counter()
                    self.get_logger().info(f"visualize + publish path took {(t5 - t4) * 1000:.1f} ms")
        else:
            if self.obs_img is not None:
                self.get_logger().info(f"waiting on camera")
        waypoint_msg = Float32MultiArray()
        waypoint_msg.data = chosen_waypoint.flatten().tolist()
        self.waypoint_pub.publish(waypoint_msg)

        self.inference_count += 1
        elapsed = time.perf_counter() - self.inference_start_time

        if elapsed >= 1.0:
            inference_rate = self.inference_count / elapsed
            if self.obs_img is not None:
                self.get_logger().info(
                    f"Inference rate: {inference_rate:.2f} Hz "
                    f"({self.inference_count} in {elapsed:.2f}s)"
                )

            self.inference_count = 0
            self.inference_start_time = time.perf_counter()
        # print(f"image queue {len(self.image_queue)} chosen waypoint: {chosen_waypoint}")

        # reached_goal = self.closest_node == self.goal_node
        # goal_reached_msg = Bool()
        # goal_reached_msg.data = bool(reached_goal)
        # self.goal_pub.publish(goal_reached_msg)

        # if reached_goal:
        #     print("Reached goal! Stopping...")

def main(args: argparse.Namespace):
    rclpy.init()
    navila_node = NaVILANode(args)

    try:
        rclpy.spin(navila_node)
    except KeyboardInterrupt:
        pass
    finally:
        navila_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ros inference pipeline for NaVILA."
    )
    parser.add_argument("-r", "--robot", type=str, help="Robot Name", default="husky")
    parser.add_argument("-i", "--instruction", type=str, help="instruction", default="go forward")
    parser.add_argument("-s", "--scale", type=int, help="waypoint scale factor int", default=4)
    args = parser.parse_args()
    main(args)
