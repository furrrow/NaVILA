from __future__ import annotations
import time
import argparse
import cv2
import numpy as np
from PIL import Image
import torch
# import matplotlib
# matplotlib.use("TkAgg")
from NaVILA_inference import NavilaPolicy, navila_command_to_waypoints

from custom_utils.stream_handler import FrameStatus, InputStreamHandler
from custom_utils.io_utils import save_depth_video_mp4, overlay_path
from custom_utils.io_utils import load_calibration, filter_unwanted_results

def transform_point(T_base_from_cam: np.ndarray, point_cam: np.ndarray) -> np.ndarray:
    point_h = np.append(point_cam, 1.0)   # [x, y, z, 1]
    point_base_h = T_base_from_cam @ point_h
    return point_base_h[:3]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="pipeline to show omnivla."
    )
    parser.add_argument("--robot-radius", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Initialize predictor (single-GPU streaming)
    save_video_toggle = True
    time_session = False
    stream_type = "video"  # ["yarp", "video", "webcam"]
    video_path = "/home/jim/Projects/steernav/assets/corridoor_omni_ft_2_left.mp4"
    video_path = "/home/jim/Projects/steernav/assets/close_up_bug_20260909_114902.mp4"
    camera_matrix_dir = "/home/jim/Projects/steernav/steernav/cam_matrix.json"
    model_path = "./checkpoints/navila-llama3-8b-8f"
    instruction = "go to the kitchen"
    output_folder = "demo_video"
    webcam_index = 0

    policy = NavilaPolicy(model_path)
    print("loading model: ", model_path)

    cam_matrix, dist_coeffs, T_base_from_cam = load_calibration(camera_matrix_dir)
    T_cam_from_base = np.linalg.inv(T_base_from_cam)

    # Initialize input source
    src = InputStreamHandler(
        kind=stream_type,
        video_path=video_path,
        webcam_index=webcam_index,
        skip_n_fr=10,
    )
    print(f"Opening source: {stream_type}")
    src.open()
    stream_buffer = src.read()

    peak_memory = 0
    frame_idx = 0

    frame_timestamps = []  # To compute output fps
    video_frames = []  # Buffer of frames for final video save
    stop_processing = False
    # shrink image dimensions
    img_w, img_h = 640, 480
    window_name = "esdf_surface"
    # Create a resizable window
    # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Set fixed width x height (e.g., 640x480 or 1280x720)
    # cv2.resizeWindow(window_name, 720, 720)

    prev_time = time.time()
    try:
        while stop_processing is not True:
            # Read frame (RGB)
            stream_buffer = src.read()
            if stream_buffer.status == FrameStatus.NO_FRAME:
                # YARP: no new frame yet; try again.
                continue
            if stream_buffer.status == FrameStatus.EOS:
                # End of stream for video/webcam or closed YARP port.
                break
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            original_frame = stream_buffer.frame
            # shrink frame for faster inference
            # frame_rgb = cv2.resize(original_frame, dsize=(img_w, img_h), interpolation=cv2.INTER_CUBIC)
            frame_rgb = original_frame
            pil_image = Image.fromarray(frame_rgb)
            t0 = time.perf_counter()

            policy.add_frame(pil_image)
            output = policy.predict(instruction)
            print(output)
            waypoints = navila_command_to_waypoints(output)
            path_xy = waypoints[:, :2]
            # print(path_xy)

            t0 = time.perf_counter()
            overlay_img = overlay_path(trajectories=path_xy,
                                        img=frame_rgb,
                                        cam_matrix=cam_matrix,
                                        T_cam_from_base=T_cam_from_base,)
            t1 = time.perf_counter()
            # print(f"visualize_path {(t1 - t0) * 1000:.1f} ms")
            # Display FPS
            cv2.putText(overlay_img,f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1,(0, 255, 0),2)
            # cv2.imshow(
            #     window_name, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
            # )
            # cv2.waitKey(1) # cv2.waitKey(1) if running in a real-time loop
            # Update running statistics
            frame_idx += 1
            frame_timestamps.append(time.time())
            if save_video_toggle:
                video_frames.append(cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
            current_peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 3  # GB
            peak_memory = max(peak_memory, current_peak_memory)
            print(
                f"Processed frame {frame_idx}. "
                f"Current peak memory: {current_peak_memory:.2f} GB, "
                f"Overall peak memory: {peak_memory:.2f} GB.",
                end="\r",
            )
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, stopping processing gracefully...")

    finally:
        # Source cleanup
        src.close()

        if save_video_toggle:
            if len(frame_timestamps) >= 2:
                elapsed = frame_timestamps[-1] - frame_timestamps[0]
                # Use average FPS over the whole run
                effective_fps = (len(frame_timestamps) - 1) / elapsed
            else:
                effective_fps = 30.0

            output_dir = f"{output_folder}/{stream_type}.mp4"
            save_depth_video_mp4(
                video=np.array(video_frames),
                path=output_dir,
                fps=4,
                # fps=effective_fps,
            )
            print(
                f"\nSaved video to {output_dir} at {effective_fps:.2f} FPS."
            )

        # Close any OpenCV windows
        cv2.destroyAllWindows()

        print(f"Processed {frame_idx} frames.")
        print(f"Peak GPU memory usage: {peak_memory:.2f} GB.")


if __name__ == "__main__":
    raise SystemExit(main())
