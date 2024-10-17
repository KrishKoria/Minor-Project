import queue
import cv2
import numpy as np
import time
from ultralytics import YOLO
import threading


def generate_colors(num_classes):
    return [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(num_classes)]


def capture_frames(cap, frame_queue, frame_skip, capture_times):
    frame_count = 0
    while True:
        start_capture = time.time()
        ret, frame = cap.read()
        if not ret:
            break
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


def detect_objects_from_webcam():
    # Use a smaller, more efficient model
    yolo_model = YOLO('best.pt')

    # Use CPU for Raspberry Pi (most don't have CUDA)
    device = 'cpu'
    yolo_model.to(device)

    # Export to ONNX format for better performance on CPU
    yolo_model.export(format="onnx", device=device)
    yolo_model = YOLO('best.onnx')

    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Reduced resolution
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # Reduced resolution

    prev_time = 0
    frame_skip = 3  # Increased frame skip
    frame_queue = queue.Queue(maxsize=1)

    capture_times = []
    process_times = []
    fps_values = []

    capture_thread = threading.Thread(target=capture_frames,
                                      args=(video_capture, frame_queue, frame_skip, capture_times))
    capture_thread.daemon = True
    capture_thread.start()

    colors = generate_colors(80)

    while True:
        if frame_queue.empty():
            continue

        frame = frame_queue.get()

        frame = process_frame(yolo_model, frame, colors, process_times)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        fps_values.append(fps)

        # Reduce text rendering frequency
        if len(fps_values) % 10 == 0:
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Webcam Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
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
    detect_objects_from_webcam()