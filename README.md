# Object Detection Benchmarking on Raspberry Pi

This project benchmarks object detection models on a Raspberry Pi using different frameworks and formats, including PyTorch, ONNX, and NCNN. The goal is to find the most efficient model for object detection on a Raspberry Pi.

## Project Information

This project was created for the 5th Semester B. Tech Minor Project at the Department of Computer Science and Engineering,

## Files

- `benchmark_rpi_benchmark.py`: Benchmarks object detection using the YOLO model on a Raspberry Pi with PyTorch.
- `benchmark_rpi_ncnn.py`: Benchmarks object detection using the YOLO model on a Raspberry Pi with NCNN.
- `benchmark_rpi_onnx.py`: Benchmarks object detection using the YOLO model on a Raspberry Pi with ONNX.
- `benchmarked-most-efficient.py`: Benchmarks the most efficient object detection model.
- `best_ncnn_model/`: Contains the NCNN model files and metadata.
  - `metadata.yaml`: Metadata for the NCNN model.
  - `model_ncnn.py`: Script for testing NCNN model inference.
  - `model.ncnn.param`: NCNN model parameters.
- `best.onnx`: The ONNX model file.
- `best.pt`: The PyTorch model file.
- `best.torchscript`: The TorchScript model file.
- `README.md`: This README file.

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- PyTorch
- Ultralytics YOLO
- NCNN
- Picamera2 (for Raspberry Pi camera)

## Installation

1.  Clone the repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run the benchmarking scripts:
   - For PyTorch:
     ```sh
     python benchmark_rpi_benchmark.py
     ```
   - For NCNN:
     ```sh
     python benchmark_rpi_ncnn.py
     ```
   - For ONNX:
     ```sh
     python benchmark_rpi_onnx.py
     ```
2. Run the script to benchmark the most efficient model:
   ```sh
   python benchmarked-most-efficient.py
   ```
