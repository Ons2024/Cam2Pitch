import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv
from sports.configs.soccer import SoccerPitchConfiguration
from sports.annotators.soccer import draw_pitch
from sports.common.view import ViewTransformer

# Environment Configuration
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CUDAExecutionProvider]"

# Constants
VIDEO_INPUT_PATH = "data/new1.mp4"  # Path to input video
VIDEO_OUTPUT_PATH = "output_video.mp4"  # Path to save the output video
PITCH_DETECTION_MODEL_PATH = 'models/best.pt'

# Soccer Pitch Configuration
CONFIG = SoccerPitchConfiguration()

# Model Initialization (YOLO model for field detection)
FIELD_DETECTION_MODEL = YOLO(PITCH_DETECTION_MODEL_PATH).to(device="cuda")

def get_homography_matrix(frame):
    """
    Detects the soccer field in the frame and returns the homography matrix.

    Args:
        frame (np.ndarray): The current video frame.

    Returns:
        tuple: Homography matrix and the transformer object.
    """
    pitch_result = FIELD_DETECTION_MODEL(frame, verbose=False)[0]
    key_points = sv.KeyPoints.from_ultralytics(pitch_result)

    # Ensure keypoints are detected
    if len(key_points.xy) == 0 or len(key_points.xy[0]) == 0:
        raise ValueError("No keypoints detected in the frame. Ensure the pitch is visible.")

    filter_mask = key_points.confidence[0] > 0.7
    frame_reference_points = key_points.xy[0][filter_mask]
    pitch_reference_points = np.array(CONFIG.vertices)[filter_mask]

    # Ensure enough keypoints are detected for homography
    if len(frame_reference_points) < 4:
        raise ValueError(
            f"Not enough reliable keypoints detected ({len(frame_reference_points)}). "
            f"Need at least 4 for homography. Try adjusting the confidence threshold."
        )
    
    transformer = ViewTransformer(
        source=frame_reference_points,
        target=pitch_reference_points
    )
    return transformer.m, transformer

def process_video(video_path, output_path):
    """
    Processes the input video frame by frame, applies transformations, and saves the result as a new video.

    Args:
        video_path (str): The path to the input video file.
        output_path (str): The path to save the output video.
    """
    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video at {video_path}")

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define codec and create VideoWriter object for saving output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Generate soccer pitch visualization (static background)
    annotated_pitch_image = draw_pitch(CONFIG)

    # Process each frame of the video
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Processing Video Frames")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        try:
            # Compute homography matrix for current frame
            homography_matrix, transformer = get_homography_matrix(frame)

            # Transform the current frame to match the pitch dimensions
            output_resolution = (CONFIG.length, CONFIG.width)
            transformed_image = transformer.transform_image(frame, output_resolution)

            # Create a binary mask for the transformed image area
            _, binary_mask = cv2.threshold(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)

            # Resize binary mask to match pitch dimensions
            resized_mask = cv2.resize(binary_mask, (annotated_pitch_image.shape[1], annotated_pitch_image.shape[0]))

            # Create a gray overlay image
            gray_color = (128, 128, 128)  # Gray color in BGR
            gray_overlay = np.full(annotated_pitch_image.shape, gray_color, dtype=np.uint8)

            # Apply the mask to the gray overlay
            masked_gray_overlay = cv2.bitwise_and(gray_overlay, gray_overlay, mask=resized_mask)

            # Blend the masked gray overlay with the original pitch image
            alpha = 0.5  # Adjust for desired transparency
            blended_image = cv2.addWeighted(masked_gray_overlay, alpha, annotated_pitch_image, 1 - alpha, 0)

            # Resize blended image to match original frame dimensions (for saving)
            resized_blended_image = cv2.resize(blended_image, (frame_width, frame_height))

            # Write processed frame to output video file
            out.write(resized_blended_image)
        
        except ValueError as e:
            print(f"Error processing frame: {e}")
        
        pbar.update(1)
    
    pbar.close()
    
    # Release resources
    cap.release()
    out.release()

if __name__ == "__main__":
    try:
        process_video(VIDEO_INPUT_PATH, VIDEO_OUTPUT_PATH)
        print(f"Video processing complete! Output saved at: {VIDEO_OUTPUT_PATH}")
    
        # Display a sample processed frame from the output video (optional)
        cap_out = cv2.VideoCapture(VIDEO_OUTPUT_PATH)
        ret_out, sample_frame_out = cap_out.read()
        if ret_out:
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(sample_frame_out, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.title("Sample Processed Frame")
            plt.show()
        cap_out.release()
    
    except Exception as e:
        print(f"An error occurred: {e}")
