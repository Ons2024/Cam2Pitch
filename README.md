# Cam2Pitch: Soccer Field Transformation

Cam2Pitch is a computer vision project that leverages YOLO object detection to identify and transform soccer fields in videos. The project processes video frames, detects the field, and overlays a semi-transparent pitch on top of the detected field, generating a new output video.

## Features

- **Field Detection**: Uses YOLO object detection to identify soccer fields in video frames.
- **Pitch Overlay**: Applies a semi-transparent pitch overlay on the detected field.
- **Video Processing**: Outputs a transformed video with the overlaid pitch.

## Steps Involved:

1. **Dataset Creation**:
   - A custom dataset was created by detecting the corners (keypoints) of the soccer field in various images and videos.
   - These keypoints were manually annotated or extracted using corner detection techniques.
   - The dataset was then paired with a synthetic soccer field model, ensuring that the detected keypoints matched the corners of the synthetic field.
   
2. **Keypoint Detection**:
   - A deep learning model was trained to detect the keypoints on the soccer field in images. This model learned to match the keypoints detected in the real-world video frames to the synthetic field model.
   
3. **Homography Generation**:
   - Using the detected keypoints in the real-world video, the model computes a **homography matrix** to map the camera view of the soccer field to the synthetic field.
   - This homography matrix is used to transform the field in the video to fit the dimensions and view of the synthetic field.

4. **Field Transformation**:
   - After applying the homography, the video frame is transformed to align with the synthetic field.
   - A semi-transparent overlay of the synthetic field is applied to the transformed video, creating a visual effect where the soccer field appears to be aligned with the synthetic pitch.


## Installation

To get started with Cam2Pitch, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/Cam2Pitch.git
    cd Cam2Pitch
    ```

2. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```


## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

## Acknowledgements

This project incorporates portions of code from the [Roboflow](https://github.com/roboflow) project, which is also licensed under the MIT License. Special thanks to the Roboflow team for their contributions to the computer vision community.

## Contributing

Contributions are welcome! If you'd like to improve this project, feel free to fork the repository, make your changes, and submit a pull request.


