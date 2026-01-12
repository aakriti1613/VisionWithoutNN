import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple


class _Track:
    """Lightweight track for associating detections across frames."""

    def __init__(self, track_id: int, centroid: Tuple[float, float], bbox: Tuple[int, int, int, int]) -> None:
        self.track_id = track_id
        self.centroid = np.array(centroid, dtype=np.float32)
        self.bbox = bbox  # (x, y, w, h)
        self.centroid_history: Deque[Tuple[float, float]] = deque([centroid], maxlen=32)
        self.frames_since_seen = 0
        self.counted = False
        self.initial_centroid = centroid

    def update(self, centroid: Tuple[float, float], bbox: Tuple[int, int, int, int]) -> None:
        """Update centroid and bounding box for the track."""
        self.centroid = np.array(centroid, dtype=np.float32)
        self.bbox = bbox
        self.centroid_history.append(centroid)
        self.frames_since_seen = 0

    def mark_missed(self) -> None:
        """Increase the number of frames the track has been missed."""
        self.frames_since_seen += 1


@dataclass
class _SceneCalibration:
    min_area: float
    min_height: float
    counting_line: float
    min_path_length: float
    roi_left: float
    roi_right: float


class Solution:
    """Vehicle counter using classical computer vision and motion analysis."""

    VIDEO_EXTENSIONS: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpg", ".mpeg")

    def __init__(self, debug: bool = False) -> None:
        # Parameters tuned to work across a variety of static traffic scenes
        self.debug = debug
        self.bg_history = 500
        self.bg_threshold = 25
        self.max_tracking_distance_ratio = 0.08  # relative to frame diagonal
        self.max_track_age = 30  # frames
        self.min_path_ratio = 0.02  # relative movement threshold for counting
        self.min_area_ratio = 0.00009  # relative to frame area
        self.min_height_ratio = 0.026
        self.aspect_ratio_range = (0.15, 6.0)
        self.min_history_for_count = 3
        self.min_history_for_fallback = 5
        self.calibration_frames = 180

    def forward(self, video_path: str) -> int:
        """
        Count the number of vehicles moving away from the camera.

        Args:
            video_path: Path to a single video file or a directory containing videos.

        Returns:
            Total number of unique vehicles detected. When a directory is provided
            the returned value is the sum across all discovered videos.
        """
        path = Path(video_path)
        if not path.exists():
            raise ValueError(f"Video path does not exist: {video_path}")

        if path.is_dir():
            # Sum counts across every supported video file in the directory (recursively).
            per_video_counts = self.forward_directory(video_path)
            return sum(per_video_counts.values())

        if not path.is_file():
            raise ValueError(f"Path is neither a file nor directory: {video_path}")

        return self._process_single_video(path)

    def forward_directory(self, directory_path: str) -> Dict[str, int]:
        """
        Process every supported video inside a directory (recursively).

        Args:
            directory_path: Directory that houses the dataset videos.

        Returns:
            Mapping of relative video paths to their individual counts.
        """
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Directory does not exist: {directory_path}")

        video_files = self._iter_video_files(directory)
        if not video_files:
            raise ValueError(f"No supported video files found in: {directory_path}")

        results: Dict[str, int] = {}
        for file_path in video_files:
            relative_key = str(file_path.relative_to(directory))
            results[relative_key] = self._process_single_video(file_path)

        return results

    def _iter_video_files(self, directory: Path) -> List[Path]:
        """Return all supported video files under a directory, sorted for stability."""
        video_files = [
            path for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in self.VIDEO_EXTENSIONS
        ]
        return sorted(video_files)

    def _process_single_video(self, video_path: Path) -> int:
        """Run the vehicle counting pipeline for a single video file."""
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        ret, frame = capture.read()
        if not ret:
            capture.release()
            return 0

        frame_height, frame_width = frame.shape[:2]
        frame_area = float(frame_height * frame_width)
        frame_diagonal = float(np.hypot(frame_width, frame_height))

        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.bg_history,
            varThreshold=self.bg_threshold,
            detectShadows=True,
        )

        # Pre-compute ROI mask to suppress peripheral noise (e.g., pedestrians on sidewalks)
        roi_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        roi_top = 0
        roi_bottom = frame_height
        roi_left = 0
        roi_right = frame_width
        roi_mask[roi_top:roi_bottom, roi_left:roi_right] = 255

        counting_line = int(frame_height * 0.4)
        min_area = max(int(frame_area * self.min_area_ratio), 100)
        min_height = max(int(frame_height * self.min_height_ratio), 24)
        max_tracking_distance = frame_diagonal * self.max_tracking_distance_ratio
        min_path_length = frame_height * self.min_path_ratio

        calibration = self._calibrate_scene(
            video_path,
            frame_height=frame_height,
            frame_width=frame_width,
            roi_mask=roi_mask,
        )
        if calibration:
            roi_left = int(np.clip(calibration.roi_left, 0, frame_width - 1))
            roi_right = int(np.clip(calibration.roi_right, roi_left + 1, frame_width))
            roi_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
            roi_mask[roi_top:roi_bottom, roi_left:roi_right] = 255

            min_area = int(
                np.clip(
                    calibration.min_area,
                    min_area * 0.5,
                    max(min_area * 1.5, min_area + 50),
                )
            )
            min_area = max(min_area, 60)
            min_height = int(
                np.clip(
                    calibration.min_height,
                    max(min_height * 0.5, 18),
                    max(min_height * 1.5, min_height + 10),
                )
            )
            min_height = max(min_height, 18)
            counting_line = int(
                np.clip(
                    calibration.counting_line,
                    frame_height * 0.25,
                    frame_height * 0.6,
                )
            )
            min_path_length = max(min_path_length * 0.5, calibration.min_path_length)

        tracks: List[_Track] = []
        next_track_id = 1
        total_count = 0

        # Morphology kernels reused for efficiency
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # Rewind capture to process from the first frame
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ret, frame = capture.read()
            if not ret:
                break

            fg_mask = bg_subtractor.apply(frame)
            if fg_mask is None:
                continue

            # Remove shadows (pixel value < 200) and apply ROI
            _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
            fg_mask = cv2.bitwise_and(fg_mask, roi_mask)

            # Noise reduction and blob consolidation
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
            fg_mask = cv2.dilate(fg_mask, dilate_kernel, iterations=1)

            # Extract candidate vehicle regions
            contours_info = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

            detections: List[Tuple[Tuple[float, float], Tuple[int, int, int, int]]] = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                if h < min_height:
                    continue

                aspect_ratio = w / float(h) if h > 0 else 0.0
                if not (self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]):
                    continue

                centroid = (x + w / 2.0, y + h / 2.0)
                detections.append((centroid, (x, y, w, h)))

            used_tracks = set()
            used_detections = set()

            # Associate detections with existing tracks
            if detections:
                distances = []
                for track_idx, track in enumerate(tracks):
                    for detection_idx, (centroid, _) in enumerate(detections):
                        dist = np.linalg.norm(track.centroid - np.array(centroid, dtype=np.float32))
                        distances.append((dist, track_idx, detection_idx))
                distances.sort(key=lambda item: item[0])

                for dist, track_idx, detection_idx in distances:
                    if dist > max_tracking_distance:
                        continue
                    if track_idx in used_tracks or detection_idx in used_detections:
                        continue
                    used_tracks.add(track_idx)
                    used_detections.add(detection_idx)
                    centroid, bbox = detections[detection_idx]
                    tracks[track_idx].update(centroid, bbox)

            # Increase age for unmatched tracks
            for track_index, track in enumerate(tracks):
                if track_index in used_tracks:
                    continue
                track.mark_missed()

            # Unmatched detections start new tracks
            if detections:
                for detection_idx, (centroid, bbox) in enumerate(detections):
                    if detection_idx in used_detections:
                        continue
                    new_track = _Track(next_track_id, centroid, bbox)
                    tracks.append(new_track)
                    next_track_id += 1

            # Count vehicles crossing the counting line
            for track in tracks:
                if track.counted or len(track.centroid_history) < self.min_history_for_count:
                    continue

                previous_y = track.centroid_history[-2][1]
                current_y = track.centroid_history[-1][1]

                if previous_y >= counting_line > current_y and previous_y > current_y:
                    displacement = track.initial_centroid[1] - current_y
                    if displacement >= min_path_length:
                        track.counted = True
                        total_count += 1
                        if self.debug:
                            print(
                                f"[COUNT] track={track.track_id} frames={len(track.centroid_history)} "
                                f"disp={displacement:.1f} line_y={counting_line} "
                                f"centroid={track.centroid}"
                            )

            # Remove stale tracks and perform a final eligibility check before dropping them
            active_tracks: List[_Track] = []
            for track in tracks:
                if track.frames_since_seen <= self.max_track_age:
                    active_tracks.append(track)
                    continue

                if track.counted or not track.centroid_history:
                    continue

                final_y = track.centroid_history[-1][1]
                displacement = track.initial_centroid[1] - final_y
                crossed_line = any(
                    prev[1] >= counting_line > curr[1]
                    for prev, curr in zip(track.centroid_history, list(track.centroid_history)[1:])
                )
                if (
                    track.initial_centroid[1] > counting_line
                    and displacement >= min_path_length
                    and final_y <= counting_line
                    and len(track.centroid_history) >= self.min_history_for_fallback
                    and crossed_line
                ):
                    track.counted = True
                    total_count += 1
                    if self.debug:
                        print(
                            f"[COUNT-FALLBACK] track={track.track_id} frames={len(track.centroid_history)} "
                            f"disp={displacement:.1f} line_y={counting_line} final_y={final_y:.1f}"
                        )

            tracks = active_tracks

        capture.release()
        return int(total_count)

    def _calibrate_scene(
        self,
        video_path: Path,
        frame_height: int,
        frame_width: int,
        roi_mask: np.ndarray,
    ) -> Optional[_SceneCalibration]:
        """Derive scene-specific thresholds from a short warm-up pass."""
        calib_capture = cv2.VideoCapture(str(video_path))
        if not calib_capture.isOpened():
            return None

        calib_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=max(100, self.bg_history // 2),
            varThreshold=self.bg_threshold,
            detectShadows=True,
        )

        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        min_raw_area = max(int(frame_height * frame_width * 0.00005), 60)
        min_raw_height = max(int(frame_height * 0.02), 15)

        areas: List[float] = []
        heights: List[float] = []
        centroid_ys: List[float] = []
        centroid_xs: List[float] = []

        max_frames = int(
            min(
                calib_capture.get(cv2.CAP_PROP_FRAME_COUNT) or self.calibration_frames,
                self.calibration_frames,
            )
        )
        frames_processed = 0

        while frames_processed < max_frames:
            ret, frame = calib_capture.read()
            if not ret:
                break
            frames_processed += 1

            fg_mask = calib_subtractor.apply(frame)
            if fg_mask is None:
                continue

            _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
            fg_mask = cv2.bitwise_and(fg_mask, roi_mask)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

            contours_info = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_raw_area:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                if h < min_raw_height:
                    continue

                aspect_ratio = w / float(h) if h > 0 else 0.0
                if not (0.1 <= aspect_ratio <= 8.0):
                    continue

                areas.append(area)
                heights.append(float(h))
                centroid_ys.append(y + h / 2.0)
                centroid_xs.append(x + w / 2.0)

        calib_capture.release()

        if len(heights) < 5:
            return None

        areas_np = np.array(areas)
        heights_np = np.array(heights)
        centroid_np = np.array(centroid_ys)

        min_area = float(np.percentile(areas_np, 30) * 0.85)
        min_area = max(min_area, float(min_raw_area))

        min_height = float(np.percentile(heights_np, 25) * 0.9)
        min_height = max(min_height, float(min_raw_height))

        median_height = float(np.percentile(heights_np, 55))
        min_path_length = max(frame_height * 0.012, median_height * 0.85)

        counting_line = float(
            np.clip(
                np.percentile(centroid_np, 60),
                frame_height * 0.25,
                frame_height * 0.6,
            )
        )

        centroid_x_np = np.array(centroid_xs)
        roi_left = float(
            np.clip(
                np.percentile(centroid_x_np, 5) - frame_width * 0.05,
                0,
                frame_width - 1,
            )
        )
        roi_right = float(
            np.clip(
                np.percentile(centroid_x_np, 95) + frame_width * 0.05,
                roi_left + 1,
                frame_width,
            )
        )

        return _SceneCalibration(
            min_area=min_area,
            min_height=min_height,
            counting_line=counting_line,
            min_path_length=min_path_length,
            roi_left=roi_left,
            roi_right=roi_right,
        )

if __name__ == "__main__":
    import sys

    try:
        raw_path = input(
            "Enter the full path to a traffic video file or a directory containing videos:\n> "
        ).strip()
    except EOFError:
        print("Error: No input received.")
        sys.exit(1)

    if not raw_path:
        print("Error: Empty path provided.")
        sys.exit(1)

    # Allow users to paste paths wrapped in quotes.
    raw_path = raw_path.strip("\"'")
    target = Path(raw_path)

    solution = Solution()
    try:
        if target.is_dir():
            counts = solution.forward_directory(str(target))
            for relative_path, count in counts.items():
                print(f"{relative_path}: {count}")
            print(f"Total vehicles across dataset: {sum(counts.values())}")
        else:
            total = solution.forward(str(target))
            print(f"Vehicles detected: {total}")
    except ValueError as exc:
        print(f"Error: {exc}")