# 3D Coordinate System in Pose Estimation

This document explains the 3D coordinate system used in this pose estimation project.

## Coordinate System

When using a standard RGB camera (without depth information):

-   X: Horizontal pixel position (increases from left to right)
-   Y: Vertical pixel position (increases from top to bottom)
-   Z: Always 0 (since depth information is not available)

When using the RealSense D435i depth camera:

-   X: Horizontal pixel position (increases from left to right)
-   Y: Vertical pixel position (increases from top to bottom)
-   Z: Depth value in meters (distance from the camera)

## RealSense Depth Camera Details

The RealSense D435i provides depth information that allows for true 3D coordinates. The camera uses an infrared projector and stereo cameras to calculate depth information for each pixel.

### Camera Specifications:

-   Field of View: 69° × 42° (H × V)
-   Depth Technology: Active IR Stereo
-   Depth Range: ~0.3m to ~10m
-   Depth Resolution: Up to 1280 × 720
-   RGB Resolution: Up to 1920 × 1080

### Coordinate Transformation

The system aligns the depth and color frames, so the 3D coordinates shown correspond directly to the 2D keypoints visible in the color image.

For each detected keypoint:

1. The (X, Y) coordinates are taken from the detected 2D pose
2. The Z coordinate is taken from the aligned depth map at the same (X, Y) position
3. The Z value is converted to meters using the depth scale of the camera

## Visualization

For each detected keypoint, the system displays:

```
keypoint_id: (X, Y, Z)
```

Where:

-   `keypoint_id` is the index of the keypoint (e.g., 0 for nose, 1 for left eye, etc.)
-   `X` and `Y` are in pixel coordinates
-   `Z` is in meters (or 0 if depth information is not available)

## Accuracy Considerations

-   The accuracy of Z (depth) depends on the distance to the subject:
    -   More accurate for subjects 0.5m - 4m from the camera
    -   Less accurate for very close or distant subjects
-   Well-lit environments provide better depth estimation
-   Reflective surfaces may cause inaccurate depth readings
-   The edges of objects may have less accurate depth values due to the difference in viewpoints between the stereo cameras
