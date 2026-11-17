"""
export_onnx.py  -  run ONCE on your own computer.
Converts the YOLO model to ONNX so the Render (lite) version can use it
without PyTorch. Creates yolov8n.onnx in this folder.

Run:
    py -3.12 export_onnx.py
(If it says it needs to install 'onnx', let it. It needs internet.)
"""

from ultralytics import YOLO

YOLO("yolov8n.pt").export(format="onnx", imgsz=640, opset=12)
print("Done: yolov8n.onnx created. Keep it in the Smart_parking folder.")
