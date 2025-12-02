# import cv2 as cv
# import numpy as np
# from ultralytics import YOLO
# import socket

# # --- CẤU HÌNH ---
# # Giả lập Ma trận chuyển đổi (Bạn cần làm bước Calibration để có số chính xác)
# # Đây là ma trận để biến đổi Pixel (u,v) -> Thực tế (mm)
# # Thông thường bạn sẽ lưu matrix này ra file .npy và load vào
# FAKE_MATRIX = np.array([[0.5, 0, -100], [0, 0.5, -50], [0, 0, 1]]) 

# # Khởi tạo Model
# model = YOLO('yolo11s.onnx',task= 'detect')
# camera = cv.VideoCapture('http://192.168.1.36:4747/video')

# # (Tùy chọn) Khởi tạo Socket để gửi cho Robot
# # client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# # client_socket.connect(('192.168.1.10', 5000)) # IP Robot

# def pixel_to_real_coords(u, v, matrix):
#     """Hàm chuyển đổi tọa độ Pixel sang mm dùng Ma trận Homography"""
#     point = np.array([[[u, v]]], dtype=np.float32)
#     dst = cv.perspectiveTransform(point, matrix)
#     return dst[0][0][0], dst[0][0][1] # Trả về X, Y thực tế

# while True:
#     ret, frame = camera.read()
#     if not ret:
#         break

#     # 1. NHẬN DIỆN
#     results = model(frame,imgsz=640,verbose=False)
    
#     # Vẽ khung đè lên ảnh gốc để quan sát
#     annotated_frame = results[0].plot()

#     # 2. XỬ LÝ DỮ LIỆU (QUAN TRỌNG)
#     # Duyệt qua tất cả các vật thể tìm thấy
#     for box in results[0].boxes:
#         # Lấy tọa độ khung bao: x1, y1 (góc trái trên), x2, y2 (góc phải dưới)
#         coords = box.xyxy[0].cpu().numpy()
#         x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
        
#         # Lấy Class ID và độ tin cậy
#         conf = float(box.conf[0])
#         cls = int(box.cls[0])

#         # Chỉ xử lý nếu độ tin cậy > 70%
#         if conf > 0.7:
#             # TÍNH TÂM VẬT THỂ (Pixel)
#             cx = int((x1 + x2) / 2)
#             cy = int((y1 + y2) / 2)

#             # CHUYỂN ĐỔI SANG MM (REAL WORLD)
#             real_x, real_y = pixel_to_real_coords(cx, cy, FAKE_MATRIX)

#             # Hiển thị tọa độ thực lên màn hình để debug
#             text = f"Real: X={real_x:.1f}, Y={real_y:.1f}"
#             cv.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1) # Chấm đỏ ở tâm
#             cv.putText(annotated_frame, text, (cx, cy - 10), 
#                        cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#             # 3. GỬI CHO ROBOT (Ví dụ format chuỗi: "X,Y")
#             # data_string = f"{real_x:.2f},{real_y:.2f}\r\n"
#             # client_socket.send(data_string.encode('ascii'))
#             # print(f"Sent to Robot: {data_string}")

#     cv.imshow("Vision System", annotated_frame)

#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break

# camera.release()
# cv.destroyAllWindows()
# # client_socket.close()

import cv2 as cv
import numpy as np
from ultralytics import YOLO
import time

# --- CẤU HÌNH ---
# Hãy thay thế bằng ma trận bạn tính được từ script trên
REAL_MATRIX = np.array([[-0.010271381402092671, 0.6037289735230037, -66.37709041412369], [0.6116038005303223, 0.019004637869586397, -363.73840265777085], [8.830553817842488e-05, -6.954563720680957e-05, 1.0]])

# Load model ONNX (Tối ưu cho RX 580)
model = YOLO('yolov8s.pt', task='detect')
camera = cv.VideoCapture('http://192.168.1.36:4747/video')
camera.set(cv.CAP_PROP_BUFFERSIZE, 1) # Giảm độ trễ buffer

def pixel_to_real_coords(u, v, matrix):
    point = np.array([[[u, v]]], dtype=np.float32)
    dst = cv.perspectiveTransform(point, matrix)
    return dst[0][0][0], dst[0][0][1]

frame_count = 0

while True:
    ret, frame = camera.read()
    if not ret: 
        break
    
    results = model(frame, imgsz=640, verbose=False, conf=0.6)
    annotated_frame = results[0].plot()

    for box in results[0].boxes:
        # Lấy tọa độ
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        
        # TÍNH TÂM
        cx, cy = int(((x1 + x2) / 2)), int((y1 + y2) / 2)

        # CHUYỂN ĐỔI HỆ TỌA ĐỘ
        real_x, real_y = pixel_to_real_coords(cx, cy, REAL_MATRIX)

        # Vẽ tâm và tọa độ
        cv.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)
        
        # Hiển thị tọa độ X, Y (mm) ngay cạnh vật thể
        label = f"({real_x:.0f}, {real_y:.0f})mm"
        cv.putText(annotated_frame, label, (cx + 10, cy), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # In ra console để kiểm tra
        print(f"Object Detected at: X={real_x:.2f}, Y={real_y:.2f}")

    cv.imshow("Vision System", annotated_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv.destroyAllWindows()