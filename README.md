# Realtime Weapon Detection on Nvidia Jetson Nano

This project focuses on implementing and benchmarking real-time weapon detection models on an Nvidia Jetson Nano. It explores various deep learning frameworks and model formats, including PyTorch, ONNX, and NCNN, to identify the most efficient solution for deployment on edge devices like the Jetson Nano. The scripts utilize the Ultralytics YOLO framework, often starting with a `best.pt` PyTorch model and converting it on-the-fly to the target format for benchmarking. Multithreading is employed for frame capture and processing to optimize performance.

## Project Information

This project was created for the 5th Semester B. Tech Minor Project at the Department of Computer Science and Engineering.

## Files

- `benchmark_rpi_benchmark.py`: Benchmarks a YOLO model on a device with PyTorch, specifically using `Picamera2` for video input. It loads `best.pt`, exports it to NCNN format, and then benchmarks the resulting NCNN model. Inference is run on CUDA if available, otherwise CPU. _Note: `Picamera2` is specific to Raspberry Pi; adaptation may be needed for Jetson Nano's CSI camera (e.g., using GStreamer)._
- `benchmark_rpi_ncnn.py`: Benchmarks a YOLO NCNN model using a standard webcam (`cv2.VideoCapture`). It loads `best.pt`, exports it to NCNN format, and then benchmarks this NCNN model. Inference is run on CUDA if available, otherwise CPU.
- `benchmark_rpi_onnx.py`: Benchmarks a YOLO ONNX model using a standard webcam (`cv2.VideoCapture`). It loads `best.pt`, exports it to ONNX format (optimized for CPU), and then benchmarks this ONNX model specifically on the CPU. This script uses a reduced resolution (320x240) and increased frame skipping for potentially better performance on CPU-constrained devices.
- `benchmarked-most-efficient.py`: Benchmarks the NCNN version of the YOLO model, identified as potentially the most efficient setup. It loads `best.pt`, exports it to NCNN format, and runs inference using a standard webcam. Inference is run on CUDA if available, otherwise CPU. Its core logic is similar to `benchmark_rpi_ncnn.py`.
- `best_ncnn_model/`: Contains the NCNN model files and metadata.
  - `metadata.yaml`: Metadata for the NCNN model.
  - `model_ncnn.py`: A script for testing direct inference with the NCNN model files (`model.ncnn.param` and `model.ncnn.bin`). It loads the NCNN model and performs a test inference on a random input tensor.
  - `model.ncnn.bin`: NCNN model binary file.
  - `model.ncnn.param`: NCNN model parameters.
- `best.onnx`: The ONNX model file, typically generated by exporting `best.pt`.
- `best.pt`: The PyTorch model file (presumably a YOLO model). This is the base model used by benchmarking scripts for conversion.
- `best.torchscript`: The TorchScript model file.
- `README.md`: This README file.

## Requirements

- Nvidia Jetson Nano Developer Kit
- JetPack SDK (4.x or 5.x)
- Python 3.6+
- PyTorch (GPU-enabled for Jetson)
- TorchVision
- ONNX
- ONNX Runtime (GPU-enabled)
- TensorRT
- NCNN (compiled for ARM architecture, with Python bindings)
- OpenCV (pre-installed with JetPack, ensure it's built with CUDA support)
- NumPy
- Ultralytics YOLO
- CSI Camera or USB Webcam compatible with Jetson Nano
- `picamera2` (if using `benchmark_rpi_benchmark.py` as-is, primarily for Raspberry Pi)

## Installation

1. **Set up Jetson Nano:**

   - Flash your Jetson Nano with the latest JetPack SDK. Follow the official Nvidia documentation.
   - Ensure CUDA, cuDNN, and TensorRT are correctly installed as part of JetPack.

2. **Clone the repository:**

   ```sh
   git clone <repository-url>
   cd <repository-directory>
   ```

3. **Install Python dependencies:**

   - It's recommended to create a virtual environment:

     ```sh
     python3 -m venv venv
     source venv/bin/activate
     ```

   - Install PyTorch and TorchVision compatible with your JetPack version. Refer to the NVIDIA forums or official PyTorch guides for Jetson.
   - Install other packages:

     ```sh
     pip install -r requirements.txt
     # Ensure onnxruntime-gpu is installed if not in requirements.txt
     # pip install onnx onnxruntime-gpu
     # Verify NCNN Python bindings are installed (e.g., pip install ncnn)
     # If using picamera2 on a Raspberry Pi: pip install picamera2
     ```

   - Note: `requirements.txt` may need to be updated to reflect Jetson-specific versions or dependencies.

## Usage

1. **Prepare your models:**

   - Ensure your base PyTorch model (`best.pt`) is in the project root. The benchmarking scripts will typically handle the conversion to ONNX or NCNN on-the-fly.
   - Alternatively, you can pre-convert models and place `best.onnx` or the NCNN files in `best_ncnn_model/` and modify scripts if needed to load them directly.
   - The `best_ncnn_model/model_ncnn.py` script expects `model.ncnn.param` and `model.ncnn.bin` to be in the `best_ncnn_model` subdirectory relative to its execution path.

2. **Run the benchmarking scripts:**

   - The scripts will output average capture time, average processing time, and average FPS to the console.
   - Modify scripts if necessary to point to correct model paths or adjust camera parameters.
   - **For PyTorch/NCNN with PiCamera2 (Jetson adaptation may be needed):**

     - This script uses `Picamera2`. If on Jetson Nano, you might need to modify it to use a GStreamer pipeline with OpenCV for the CSI camera.

     ```sh
     python benchmark_rpi_benchmark.py
     ```

   - **For NCNN with Webcam:**

     ```sh
     python benchmark_rpi_ncnn.py
     ```

   - **For ONNX (CPU optimized) with Webcam:**

     ```sh
     python benchmark_rpi_onnx.py
     ```

3. **Run the script to benchmark the "most efficient" (NCNN) model with Webcam:**

   - This script is pre-configured to use the NCNN model, assuming it was found to be most efficient.

   ```sh
   python benchmarked-most-efficient.py
   ```

4. **Test NCNN model inference directly:**

   - Navigate to the `best_ncnn_model` directory or adjust paths in the script.

   ```sh
   python best_ncnn_model/model_ncnn.py
   ```

**Note on Camera Access on Jetson Nano:**
To use a CSI camera on Jetson Nano, you'll typically use OpenCV with a GStreamer pipeline. An example pipeline string for OpenCV's `cv2.VideoCapture()` might look like:

```python
# Example GStreamer pipeline for CSI camera on Jetson Nano
# cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, width=(int)1280, height=(int)720, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
```

Adjust the pipeline according to your camera's resolution and desired output format. For USB webcams, `cv2.VideoCapture(0)` (or other device index) usually works. The scripts using `cv2.VideoCapture(0)` should work with a USB webcam on the Jetson Nano.
