from ultralytics import YOLO

# Khuyên dùng bản nano (n) hoặc small (s) cho mượt
model = YOLO('yolo12s.pt') 

# Xuất sang định dạng ONNX
# dynamic=True giúp linh hoạt kích thước ảnh
model.export(format='onnx', dynamic=True)