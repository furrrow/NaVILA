from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path as PathMsg
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Empty, Float32MultiArray, String

from NaVILA_inference import NavilaPolicy, navila_command_to_waypoints


REPO_ROOT = Path(__file__).resolve().parent


class NaVILANode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("navila_node")
        config_path = Path(args.config).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as config_file:
            deploy_config = yaml.safe_load(config_file)
        robot_config = deploy_config[args.robot]

        self.robot_name = args.robot
        self.simulation_mode = args.robot == "sim_nova"
        self.instruction = args.instruction
        self.rate = float(deploy_config["frame_rate"])
        self.waypoint_scale_factor = int(args.scale)
        self.lookahead_distance = float(robot_config.get("lookahead_distance", 0.3))
        self.original_img_size = (int(deploy_config["img_w"]), int(deploy_config["img_h"]))
        self.shrink_img_size = (int(deploy_config["shrink_w"]), int(deploy_config["shrink_h"]))
        self.path_frame_id = "base_link"
        self.visualize = args.visualize
        self.obs_img = None
        self.obs_img_timestamp = None
        self.odom = None
        self.current_episode_id = None
        self._started_sent = False
        self.inference_count = 0
        self.inference_start_time = time.perf_counter()
        self.br = CvBridge()

        model_path = Path(args.model_path).expanduser().resolve()
        self.get_logger().info(f"Loading model from {model_path}")
        self.policy = NavilaPolicy(str(model_path))
        self.get_logger().info("Model ready.")

        image_topic = robot_config["image_topic"]
        self.compressed_img_topic = "compressed" in image_topic
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        image_type = CompressedImage if self.compressed_img_topic else Image
        self.image_sub = self.create_subscription(
            image_type, image_topic, self.img_callback_obs, sensor_qos
        )
        self.odom_sub = None
        self.episode_sub = None
        if self.simulation_mode:
            self.odom_sub = self.create_subscription(
                Odometry, robot_config["odom_topic"], self.odom_callback_obs, sensor_qos
            )
            episode_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
            )
            self.episode_sub = self.create_subscription(
                String, "/isaacsim/episode", self.episode_callback, episode_qos
            )
            self.get_logger().info("Waiting for simulation episodes...")

        self.pub_path = self.create_publisher(PathMsg, robot_config["policy_path_topic"], sensor_qos)
        self.waypoint_pub = self.create_publisher(
            Float32MultiArray, robot_config["waypoint_topic"], sensor_qos
        )
        self.pub_started = self.create_publisher(Empty, "/started", 10)

        self.cam_matrix = None
        self.T_cam_from_base = None
        self.trajectory_visual_pub = None
        if self.visualize:
            from custom_utils.io_utils import load_calibration

            calibration_path = Path(args.camera_calibration).expanduser().resolve()
            self.cam_matrix, _, T_base_from_cam = load_calibration(str(calibration_path))
            self.T_cam_from_base = np.linalg.inv(T_base_from_cam)
            self.trajectory_visual_pub = self.create_publisher(
                Image, robot_config["overlay_topic"], sensor_qos
            )

        self.timer = self.create_timer(1.0 / self.rate, self.run_inference_loop)
        self.get_logger().info("Waiting for image observations...")

    @staticmethod
    def _stamp_ns(stamp) -> int:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _reset_episode_state(self, episode_id: str, instruction: str):
        self.current_episode_id = episode_id
        self.instruction = instruction
        self.obs_img = None
        self.obs_img_timestamp = None
        self.odom = None
        self.policy.reset()
        self.inference_count = 0
        self.inference_start_time = time.perf_counter()
        self.get_logger().info(f"Reset NaVILA state for episode {episode_id}")

    def episode_callback(self, msg: String):
        try:
            episode = json.loads(msg.data)
            episode_id = episode["episode_id"]
            instruction = episode["instruction"]
            if episode.get("state") != "started":
                return
            if episode.get("robot_type") != "nova_carter":
                raise ValueError(f"unsupported robot_type {episode.get('robot_type')!r}")
            if not isinstance(episode_id, str) or not episode_id:
                raise ValueError("episode_id must be a nonempty string")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("instruction must be a nonempty string")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self.get_logger().error(f"Invalid simulation episode message: {exc}")
            return
        if episode_id != self.current_episode_id:
            self._reset_episode_state(episode_id, instruction.strip())

    def img_callback_obs(self, msg):
        if self.simulation_mode and self.obs_img_timestamp is not None:
            if self._stamp_ns(msg.header.stamp) < self._stamp_ns(self.obs_img_timestamp):
                self.get_logger().warning(
                    "Camera timestamp moved backward; clearing observation history"
                )
                self.obs_img = None
                self.odom = None
                self.policy.reset()
        if self.compressed_img_topic:
            image = self.br.compressed_imgmsg_to_cv2(msg)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = self.br.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self.obs_img_timestamp = msg.header.stamp
        self.obs_img = PILImage.fromarray(image)
        if self.obs_img.size != self.shrink_img_size:
            self.obs_img = self.obs_img.resize(self.shrink_img_size)

    def odom_callback_obs(self, msg: Odometry):
        self.odom = msg

    def _to_path_msg(self, path: np.ndarray) -> PathMsg:
        msg = PathMsg()
        msg.header.stamp = self.obs_img_timestamp
        msg.header.frame_id = self.path_frame_id
        for waypoint in path:
            x, y = waypoint[:2]
            hx, hy = waypoint[2:4] if waypoint.shape[0] >= 4 else (1.0, 0.0)
            yaw = float(np.arctan2(hy, hx))
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.z = float(np.sin(yaw / 2.0))
            pose.pose.orientation.w = float(np.cos(yaw / 2.0))
            msg.poses.append(pose)
        return msg

    def _select_waypoint(self, waypoints: np.ndarray) -> np.ndarray:
        if len(waypoints) == 1:
            return waypoints[0]
        # Keep NaVILA's existing [x, y, hx, hy] pure-rotation controller contract.
        if waypoints.shape[1] >= 4 and np.allclose(waypoints[:, :2], 0.0):
            return waypoints[-1] # return last waypoint to preserve full rotation
        segment_lengths = np.linalg.norm(np.diff(waypoints[:, :2], axis=0), axis=1)
        cumulative_lengths = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        index = int(np.searchsorted(cumulative_lengths, self.lookahead_distance, side="left"))
        index = min(index, len(waypoints) - 1)
        if len(waypoints) >= 5:
            index = min(max(index, 2), len(waypoints) - 3)
        return waypoints[index]

    def _publish_visualization(self, path_xy: np.ndarray):
        if not self.visualize or self.trajectory_visual_pub is None:
            return
        from custom_utils.io_utils import overlay_path

        overlay = overlay_path(
            trajectories=path_xy,
            img=np.array(self.obs_img.resize(self.original_img_size)),
            cam_matrix=self.cam_matrix,
            T_cam_from_base=self.T_cam_from_base,
        )
        self.trajectory_visual_pub.publish(
            self.br.cv2_to_imgmsg(np.asarray(overlay), encoding="rgb8")
        )

    def run_inference_loop(self):
        if self.obs_img is None:
            return
        if self.simulation_mode and (self.current_episode_id is None or self.odom is None):
            return

        self.policy.add_frame(self.obs_img)
        output = self.policy.predict(self.instruction)
        self.get_logger().info(f"output: {output}")
        try:
            waypoints = np.asarray(
                navila_command_to_waypoints(
                    output, num_steps=8), dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            self.get_logger().error(f"Rejecting unparseable NaVILA output: {exc}")
            return
        waypoints *= self.waypoint_scale_factor
        if (
            waypoints.ndim != 2
            or waypoints.shape[0] == 0
            or waypoints.shape[1] not in (2, 4)
        ):
            self.get_logger().error(
                f"Rejecting malformed NaVILA path with shape {waypoints.shape}"
            )
            return
        if not np.isfinite(waypoints).all():
            self.get_logger().error("Rejecting non-finite NaVILA path")
            return

        published_path = waypoints if len(waypoints) == 1 else waypoints[1:]
        chosen_waypoint = self._select_waypoint(waypoints)
        self.pub_path.publish(self._to_path_msg(published_path))
        waypoint_msg = Float32MultiArray()
        waypoint_msg.data = chosen_waypoint.tolist()
        self.waypoint_pub.publish(waypoint_msg)
        if waypoints.shape[1] == 2:
            self._publish_visualization(waypoints[:, :2])

        if not self._started_sent:
            self.pub_started.publish(Empty())
            self._started_sent = True
            self.get_logger().info("Published /started after first valid inference result.")

        self.inference_count += 1
        elapsed = time.perf_counter() - self.inference_start_time
        if elapsed >= 1.0:
            self.get_logger().info(
                f"Inference rate: {self.inference_count / elapsed:.2f} Hz "
                f"({self.inference_count} in {elapsed:.2f}s)"
            )
            self.inference_count = 0
            self.inference_start_time = time.perf_counter()


def parse_args():
    parser = argparse.ArgumentParser(description="ROS inference pipeline for NaVILA")
    parser.add_argument("-r", "--robot", default="husky")
    parser.add_argument("-i", "--instruction", default="go forward")
    parser.add_argument("-s", "--scale", type=int, default=2, help="waypoint scale factor")
    parser.add_argument("--config", default=str(REPO_ROOT / "config" / "robot.yaml"))
    parser.add_argument(
        "--model-path", default=str(REPO_ROOT / "checkpoints" / "navila-llama3-8b-8f")
    )
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--camera-calibration", default=str(REPO_ROOT / "cam_matrix.json")
    )
    args, ros_args = parser.parse_known_args()
    if ros_args and ros_args[0] != "--ros-args":
        parser.error(f"Unrecognized arguments: {ros_args}")
    if args.scale < 1:
        parser.error("--scale must be at least 1")
    return args, ros_args


def main():
    args, ros_args = parse_args()
    rclpy.init(args=ros_args)
    node = NaVILANode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
