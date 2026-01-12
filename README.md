## Vehicle Counting Solution

This document describes how the vehicle-counting module works, how to run it, and what to expect from the system. The goal is to convert raw traffic footage into a reliable estimate of the number of vehicles traveling away from the camera.

---

### 1. Problem Statement
- Input: A traffic video recorded from a static camera (or a directory of videos).
- Output: The number of distinct vehicles moving away from the camera, including cars, trucks, and two-wheelers.
- Constraints: Works with commodity hardware, relies on classical computer-vision (no GPU or heavy deep-learning model).

---

### 2. High-Level Approach
1. **Background subtraction**  
   Uses OpenCV’s MOG2 subtractor to highlight moving objects. Shadows are suppressed and a region of interest (ROI) ensures background clutter is ignored.

2. **Foreground refinement**  
   Morphological opening/closing and dilation reduce noise, bridge gaps, and produce compact blobs representing vehicles.

3. **Automatic scene calibration**  
   The first ~180 frames guide adaptive thresholds. We estimate:
   - Minimum contour area and height,
   - Effective ROI bounds (left/right crop),
   - Counting line height and minimum travel distance.  
   This keeps the counter consistent across clips with different perspectives and lane widths.

4. **Detection filtering**  
   Each contour must satisfy the calibrated size thresholds, an allowed aspect ratio, and lie inside the ROI mask.

5. **Multi-frame tracking**  
   Lightweight data association joins detections into tracks by minimizing centroid distance. Tracks persist briefly when not observed, allowing for occlusions or frame drops.

6. **Counting logic**  
   - A calibrated virtual line acts as the counting threshold.  
   - Tracks are counted when they cross the line and move far enough away from their starting position (also calibrated).  
   - Tracks that become inactive get a final eligibility check so valid vehicles aren’t dropped because of late occlusion.

7. **Directory mode**  
   When a directory is provided, every supported video (`.mp4`, `.avi`, `.mov`, `.mkv`, `.m4v`, `.mpg`, `.mpeg`) is processed recursively. The script prints per-video results and the total.

---

### 3. Key Parameters (see `main.py`)

| Parameter | Purpose | Default |
|-----------|---------|---------|
| `bg_history` | Frames history for background model | 500 |
| `bg_threshold` | Sensitivity of MOG2 detector | 25 |
| `min_area_ratio` | Minimum blob size relative to frame area | 0.00009 |
| `min_height_ratio` | Minimum blob height relative to frame height | 0.026 |
| `aspect_ratio_range` | Valid width/height ratio for vehicles | (0.15, 6.0) |
| `max_tracking_distance_ratio` | Tracker association radius relative to frame diagonal | 0.08 |
| `min_path_ratio` | Minimum displacement relative to frame height for counting | 0.02 |
| `max_track_age` | Frames before an inactive track is discarded | 30 |
| `min_history_for_count` | Minimum history length before counting | 3 frames |
| `min_history_for_fallback` | Minimum history for the fallback counter | 5 frames |
| `calibration_frames` | Frames used for auto-calibration warm-up | 180 |

These values balance sensitivity to smaller vehicles (bikes) while keeping noise manageable. The per-scene calibration clamps aggressive adjustments, so defaults remain a safe starting point. Adjust cautiously; inline code comments explain each threshold.

---

### 4. Running the Program

1. Open a terminal in `.../VisionWithoutNN`.
2. Run the script:
   ```
   py solution.py
   ```
3. When prompted, paste the full file or directory path, e.g.:
   ```
   > C:\Users\DELL\Desktop\Dataset\video_1.avi
   ```
4. The script prints either:
   - `Vehicles detected: <count>` for a single video.
   - A list of `relative/path/to/video: <count>` and the total when a directory is supplied.

**Tip:** Paths copied from File Explorer often include surrounding quotes; the script strips them automatically.

---

### 5. Interpreting Results
- Counts assume the camera is stationary and vehicles move roughly upward in the frame (away from the camera).
- Significant occlusions or erratic camera motion can reduce accuracy.
- Very close pedestrians or non-vehicle objects can occasionally be counted if they mimic vehicle motion; tighten `min_area_ratio` or `aspect_ratio_range` if needed.

---

### 6. Extending or Debugging
- Enable debug prints by constructing `Solution(debug=True)` in code; each counted track logs its track ID, frame history length, displacement, and centroid.
- For specialized scenes:
  - Override calibration outputs (ROI, counting line, min area/height) if you know the scene’s geometry.
  - Tweak morphological kernel sizes and iteration counts to balance noise removal vs. vehicle separation.
  - Lower `max_track_age` to ignore long gaps; raise it if vehicles frequently disappear behind occlusions.
- Consider persisting intermediate masks or bounding boxes with OpenCV’s drawing functions (`cv2.imshow`) when running locally.

---

### 7. Limitations & Future Improvements
- **Lighting and weather:** Heavy rain/nighttime footage may require adaptive thresholds or deep-learning-based detection.
- **Bidirectional traffic:** Vehicles moving toward the camera are not explicitly excluded if their trajectories mimic outgoing traffic.
- **Speed variance:** Very fast vehicles may appear blurred; frame rate determines how accurate the centroid history is.
- **Scalability:** Designed for offline processing. For real-time streams, integrate a frame queue and process at a throttled frame rate.

Potential enhancements include integrating a lightweight detector (YOLO-Nano/PP-YOLO Tiny), adding scene calibration for perspective-aware counting, and exporting structured logs for downstream analytics dashboards.
