import queue
import cv2
import numpy as np
import time
from ultralytics import YOLO
import threading
import torch
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput

def generate_colors(num_classes):
    return [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(num_classes)]

def capture_frames(picam2, frame_queue, frame_skip, capture_times):
    frame_count = 0
    while True:
        start_capture = time.time()
        frame = picam2.capture_array()
        capture_times.append(time.time() - start_capture)

        frame_count += 1
        if frame_count % frame_skip == 0 and not frame_queue.full():
            frame_queue.put(frame)

def process_frame(yolo_model, frame, colors, process_times):
    start_process = time.time()
    results = yolo_model(frame)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            class_id = int(box.cls[0].item())
            conf = box.conf[0].item()

            if conf >= 0.5:
                class_name = result.names[class_id]
                color = colors[class_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    process_times.append(time.time() - start_process)

    return frame

def detect_objects_from_picamera():
    yolo_model = YOLO('best.pt')
    yolo_model.to('cuda' if torch.cuda.is_available() else 'cpu')
    yolo_model.export(format="ncnn")
    yolo_model = YOLO('best_ncnn_model')

    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
    picam2.start()

    prev_time = 0
    frame_skip = 2
    frame_queue = queue.Queue(maxsize=1)

    capture_times = []  # To store capture times for benchmarking
    process_times = []  # To store processing times for benchmarking
    fps_values = []     # To store FPS values

    capture_thread = threading.Thread(target=capture_frames, args=(picam2, frame_queue, frame_skip, capture_times))
    capture_thread.daemon = True
    capture_thread.start()

    colors = generate_colors(80)

    while True:
        if frame_queue.empty():
            continue

        frame = frame_queue.get()

        start_frame = time.time()

        frame = process_frame(yolo_model, frame, colors, process_times)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        fps_values.append(fps)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Raspberry Pi Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()

    # Calculate benchmarking stats
    avg_capture_time = sum(capture_times) / len(capture_times) if capture_times else 0
    avg_process_time = sum(process_times) / len(process_times) if process_times else 0
    avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0

    print("\n--- Benchmarking Results ---")
    print(f"Average Capture Time: {avg_capture_time:.4f} seconds")
    print(f"Average Process Time: {avg_process_time:.4f} seconds")
    print(f"Average FPS: {avg_fps:.2f}")

if __name__ == "__main__":
    detect_objects_from_picamera()