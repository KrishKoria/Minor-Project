"""
Optimized Weapon Detection System with Enhanced Benchmarking

This module provides an efficient, cross-platform weapon detection system
using YOLO with various GPU and CPU optimizations for maximum performance.

Key optimizations:
- GPU: TF32, cuDNN autotuner, FP16 inference, disabled gradients, model warmup
- Threading: Event-based shutdown, proper queue handling with timeouts
- Memory: Bounded collections with deque, periodic cache clearing
- Capture: Platform-specific backends, reduced buffer latency, grab/retrieve pattern
- Inference: Confidence filtering at model level, batch data extraction
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# =============================================================================
# Detection Result Dataclass
# =============================================================================

@dataclass
class Detection:
    """Represents a single detection result with optimized memory usage."""
    __slots__ = ('x1', 'y1', 'x2', 'y2', 'class_id', 'class_name', 'confidence', 'color')

    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int
    class_name: str
    confidence: float
    color: Tuple[int, int, int]


# =============================================================================
# Performance Tracker Class
# =============================================================================

@dataclass
class PerformanceTracker:
    """
    Tracks performance metrics with rolling windows for memory efficiency.
    Uses deque with maxlen to prevent unbounded memory growth.
    """

    max_samples: int = 1000
    capture_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    process_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    fps_values: deque = field(default_factory=lambda: deque(maxlen=1000))
    total_frames: int = 0
    start_time: float = field(default_factory=time.time)
    _smoothed_fps: float = 0.0
    _fps_update_interval: int = 5  # Update displayed FPS every N frames

    def __post_init__(self):
        """Reinitialize deques with proper maxlen after dataclass init."""
        self.capture_times = deque(maxlen=self.max_samples)
        self.process_times = deque(maxlen=self.max_samples)
        self.fps_values = deque(maxlen=self.max_samples)

    def add_capture_time(self, time_sec: float) -> None:
        """Record a capture time measurement."""
        self.capture_times.append(time_sec)

    def add_process_time(self, time_sec: float) -> None:
        """Record a processing time measurement."""
        self.process_times.append(time_sec)

    def add_fps(self, fps: float) -> None:
        """Record an FPS measurement and update smoothed value."""
        self.fps_values.append(fps)
        self.total_frames += 1
        # Smooth FPS display by updating only every N frames
        if self.total_frames % self._fps_update_interval == 0:
            self._smoothed_fps = self._calculate_recent_average(self.fps_values, 10)

    def get_smoothed_fps(self) -> float:
        """Get the smoothed FPS value for display."""
        return self._smoothed_fps

    @staticmethod
    def _calculate_recent_average(data: deque, n: int = 10) -> float:
        """Calculate average of last n samples."""
        if not data:
            return 0.0
        recent = list(data)[-n:]
        return sum(recent) / len(recent)

    @staticmethod
    def _calculate_stats(data: deque) -> Dict[str, float]:
        """Calculate comprehensive statistics for a metric."""
        if not data:
            return {'avg': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0}

        arr = np.array(data)
        return {
            'avg': float(np.mean(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'std': float(np.std(arr))
        }

    def get_overall_fps(self) -> float:
        """Calculate overall FPS based on total frames and elapsed time."""
        elapsed = time.time() - self.start_time
        return self.total_frames / elapsed if elapsed > 0 else 0.0

    def print_stats(self) -> None:
        """Print comprehensive benchmarking statistics."""
        elapsed = time.time() - self.start_time

        capture_stats = self._calculate_stats(self.capture_times)
        process_stats = self._calculate_stats(self.process_times)
        fps_stats = self._calculate_stats(self.fps_values)

        print("\n" + "=" * 50)
        print("         BENCHMARKING RESULTS")
        print("=" * 50)
        print(f"\nTotal Frames Processed: {self.total_frames}")
        print(f"Total Time Elapsed: {elapsed:.2f} seconds")
        print(f"Overall FPS: {self.get_overall_fps():.2f}")

        print("\n--- Capture Time Statistics (seconds) ---")
        print(f"  Average: {capture_stats['avg']:.4f}")
        print(f"  Min:     {capture_stats['min']:.4f}")
        print(f"  Max:     {capture_stats['max']:.4f}")
        print(f"  Std Dev: {capture_stats['std']:.4f}")

        print("\n--- Processing Time Statistics (seconds) ---")
        print(f"  Average: {process_stats['avg']:.4f}")
        print(f"  Min:     {process_stats['min']:.4f}")
        print(f"  Max:     {process_stats['max']:.4f}")
        print(f"  Std Dev: {process_stats['std']:.4f}")

        print("\n--- FPS Statistics ---")
        print(f"  Average: {fps_stats['avg']:.2f}")
        print(f"  Min:     {fps_stats['min']:.2f}")
        print(f"  Max:     {fps_stats['max']:.2f}")
        print(f"  Std Dev: {fps_stats['std']:.2f}")
        print("=" * 50)


# =============================================================================
# Async Frame Capture Class
# =============================================================================

class AsyncFrameCapture:
    """
    Asynchronous frame capture with proper shutdown handling and optimizations.

    Features:
    - Event-based shutdown mechanism
    - Platform-specific camera backends
    - Frame dropping when queue is full (prefer fresh frames)
    - grab()/retrieve() pattern for frame skipping
    - Reduced buffer latency
    """

    def __init__(
        self,
        camera_id: int = 0,
        width: int = 640,
        height: int = 480,
        frame_skip: int = 2,
        queue_size: int = 2
    ):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.frame_skip = frame_skip

        self.frame_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._tracker: Optional[PerformanceTracker] = None

    @staticmethod
    def get_platform_backend() -> int:
        """Get the optimal camera backend for the current platform."""
        if sys.platform == 'win32':
            return cv2.CAP_DSHOW  # DirectShow on Windows
        elif sys.platform.startswith('linux'):
            return cv2.CAP_V4L2  # V4L2 on Linux
        else:
            return cv2.CAP_ANY  # Default backend for macOS and others

    def start(self, tracker: PerformanceTracker) -> bool:
        """Initialize camera and start capture thread."""
        self._tracker = tracker

        # Open camera with platform-specific backend
        backend = self.get_platform_backend()
        self._cap = cv2.VideoCapture(self.camera_id, backend)

        if not self._cap.isOpened():
            # Fallback to default backend
            self._cap = cv2.VideoCapture(self.camera_id)

        if not self._cap.isOpened():
            print("Error: Could not open webcam")
            return False

        # Configure camera
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency

        # Start capture thread
        self._stop_event.clear()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        return True

    def _capture_loop(self) -> None:
        """Main capture loop with frame skipping and proper shutdown handling."""
        frame_count = 0

        while not self._stop_event.is_set():
            start_capture = time.time()

            # Use grab() for frames we'll skip (faster than read())
            if not self._cap.grab():
                break

            frame_count += 1

            # Only retrieve and queue every Nth frame
            if frame_count % self.frame_skip == 0:
                ret, frame = self._cap.retrieve()
                if not ret:
                    break

                capture_time = time.time() - start_capture
                if self._tracker:
                    self._tracker.add_capture_time(capture_time)

                # Drop old frames if queue is full (prefer fresh frames)
                try:
                    # Non-blocking put - if full, clear and add new
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    pass  # Skip this frame if still can't add

    def get_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get a frame with timeout instead of busy waiting."""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        """Stop capture and release resources."""
        self._stop_event.set()

        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.0)

        if self._cap:
            self._cap.release()
            self._cap = None


# =============================================================================
# Optimized Detector Class
# =============================================================================

class OptimizedDetector:
    """
    Optimized YOLO-based weapon detector with GPU optimizations.

    Features:
    - TF32 for Ampere+ GPUs
    - cuDNN autotuner
    - Disabled gradient computation
    - FP16 (half-precision) inference support
    - Model warmup
    - Batch data extraction
    - Cached text sizes
    """

    def __init__(
        self,
        model_path: str = 'best.pt',
        confidence_threshold: float = 0.5,
        num_classes: int = 80,
        use_fp16: bool = True
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.use_fp16 = use_fp16

        # Generate colors for each class
        np.random.seed(42)  # Consistent colors across runs
        self.colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(num_classes)]

        # Text size cache for drawing operations
        self._text_size_cache: Dict[str, Tuple[Tuple[int, int], int]] = {}

        # Initialize model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model: Optional[YOLO] = None

    def initialize(self) -> None:
        """Initialize model with GPU optimizations."""
        # GPU optimizations (set before loading model)
        if self.device == 'cuda':
            # Enable TF32 for Ampere+ GPUs (RTX 30xx, 40xx)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # Enable cuDNN autotuner
            torch.backends.cudnn.benchmark = True

        # Disable gradient computation globally (inference only)
        torch.set_grad_enabled(False)

        # Load model
        self.model = YOLO(self.model_path)
        self.model.to(self.device)

        # Warmup model to avoid slow first inference
        self._warmup()

    def _warmup(self, warmup_iterations: int = 3) -> None:
        """Warm up the model with dummy inference."""
        print("Warming up model...")
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        for _ in range(warmup_iterations):
            self.model(
                dummy_frame,
                conf=self.confidence_threshold,
                verbose=False,
                half=self.use_fp16 and self.device == 'cuda'
            )

        # Clear GPU cache after warmup
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        print("Model warmup complete.")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a frame with optimizations.

        Returns list of Detection objects with all data extracted in batch.
        """
        # Run inference with optimizations
        results = self.model(
            frame,
            conf=self.confidence_threshold,  # Filter at inference level
            verbose=False,  # Reduce console spam
            half=self.use_fp16 and self.device == 'cuda'  # FP16 inference
        )

        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            # Batch extract all data at once (single CPU/GPU transfer)
            xyxy = boxes.xyxy.cpu().numpy().astype(int)
            class_ids = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            # Process all boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = xyxy[i]
                class_id = class_ids[i]
                conf = confs[i]

                detection = Detection(
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                    class_id=int(class_id),
                    class_name=result.names[class_id],
                    confidence=float(conf),
                    color=self.colors[class_id % len(self.colors)]
                )
                detections.append(detection)

        return detections

    def _get_text_size(self, text: str, font_scale: float = 0.5, thickness: int = 1) -> Tuple[Tuple[int, int], int]:
        """Get cached text size for drawing operations."""
        cache_key = f"{text}_{font_scale}_{thickness}"
        if cache_key not in self._text_size_cache:
            size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            self._text_size_cache[cache_key] = (size, baseline)
        return self._text_size_cache[cache_key]

    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw detection boxes and labels with improved visibility."""
        for det in detections:
            # Draw bounding box
            cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), det.color, 2)

            # Prepare label
            label = f"{det.class_name} {det.confidence:.2f}"
            (text_w, text_h), baseline = self._get_text_size(label)

            # Draw background rectangle for better text visibility
            label_y = max(det.y1 - 5, text_h + 5)
            cv2.rectangle(
                frame,
                (det.x1, label_y - text_h - 5),
                (det.x1 + text_w + 4, label_y + baseline - 2),
                det.color,
                -1  # Filled rectangle
            )

            # Draw text with contrasting color
            cv2.putText(
                frame,
                label,
                (det.x1 + 2, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # White text
                1,
                cv2.LINE_AA
            )

        return frame

    def clear_gpu_cache(self) -> None:
        """Periodically clear GPU cache to prevent memory buildup."""
        if self.device == 'cuda':
            torch.cuda.empty_cache()


# =============================================================================
# Main Detection Function
# =============================================================================

def draw_overlay(
    frame: np.ndarray,
    fps: float,
    object_count: int,
    show_stats_hint: bool = True
) -> np.ndarray:
    """Draw performance overlay with improved visibility."""
    height = frame.shape[0]

    # FPS display with background
    fps_text = f"FPS: {fps:.1f}"
    (text_w, text_h), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (5, 5), (text_w + 15, text_h + 15), (0, 0, 0), -1)
    cv2.putText(frame, fps_text, (10, text_h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Object count with background
    count_text = f"Objects: {object_count}"
    (text_w2, text_h2), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (5, text_h + 20), (text_w2 + 15, text_h + text_h2 + 30), (0, 0, 0), -1)
    cv2.putText(frame, count_text, (10, text_h + text_h2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Instructions hint
    if show_stats_hint:
        hint_text = "Press 'S' for stats, 'Q' to quit"
        (hint_w, hint_h), _ = cv2.getTextSize(hint_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(frame, (5, height - hint_h - 15), (hint_w + 15, height - 5), (0, 0, 0), -1)
        cv2.putText(frame, hint_text, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    return frame


def detect_weapons_from_webcam(
    camera_id: int = 0,
    model_path: str = 'best.pt',
    confidence_threshold: float = 0.5,
    frame_skip: int = 2,
    use_fp16: bool = True,
    cache_clear_interval: int = 100
) -> None:
    """
    Main weapon detection function with all optimizations.

    Args:
        camera_id: Camera device ID
        model_path: Path to YOLO model weights
        confidence_threshold: Minimum confidence for detections
        frame_skip: Process every Nth frame
        use_fp16: Use FP16 inference on GPU
        cache_clear_interval: Clear GPU cache every N frames
    """
    # Initialize components
    tracker = PerformanceTracker()
    capture = AsyncFrameCapture(
        camera_id=camera_id,
        frame_skip=frame_skip
    )
    detector = OptimizedDetector(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        use_fp16=use_fp16
    )

    # Initialize detector (with GPU setup and warmup)
    detector.initialize()

    # Start frame capture
    if not capture.start(tracker):
        print("Failed to start video capture")
        return

    print("\nStarting weapon detection...")
    print(f"Device: {detector.device}")
    print(f"FP16 Inference: {use_fp16 and detector.device == 'cuda'}")
    print("Press 'Q' to quit, 'S' for detailed stats\n")

    prev_time = time.time()
    frames_since_cache_clear = 0

    try:
        while True:
            # Get frame with timeout (no busy waiting)
            frame = capture.get_frame(timeout=0.1)
            if frame is None:
                continue

            # Process frame
            start_process = time.time()
            detections = detector.detect(frame)
            process_time = time.time() - start_process
            tracker.add_process_time(process_time)

            # Draw detections
            frame = detector.draw_detections(frame, detections)

            # Calculate FPS
            curr_time = time.time()
            instant_fps = 1.0 / (curr_time - prev_time) if curr_time > prev_time else 0.0
            prev_time = curr_time
            tracker.add_fps(instant_fps)

            # Draw overlay with smoothed FPS
            frame = draw_overlay(
                frame,
                fps=tracker.get_smoothed_fps(),
                object_count=len(detections)
            )

            # Display frame
            cv2.imshow("Weapon Detection - Optimized", frame)

            # Periodic GPU cache clearing
            frames_since_cache_clear += 1
            if frames_since_cache_clear >= cache_clear_interval:
                detector.clear_gpu_cache()
                frames_since_cache_clear = 0

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('s') or key == ord('S'):
                tracker.print_stats()

    finally:
        # Cleanup
        capture.stop()
        cv2.destroyAllWindows()

        # Print final statistics
        tracker.print_stats()


if __name__ == "__main__":
    detect_weapons_from_webcam()
