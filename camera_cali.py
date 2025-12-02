# import cv2
# import numpy as np

# # 1. Nhập tọa độ Pixel bạn nhìn thấy trên Camera (Lấy chuột click hoặc soi)
# # Thứ tự: [Góc Trái-Trên], [Góc Phải-Trên], [Góc Trái-Dưới], [Góc Phải-Dưới]
# pts_src = np.float32([[156, 120], [480, 125], [160, 400], [490, 410]])

# # 2. Nhập tọa độ Thực tế tương ứng (Đơn vị MM - Do bạn đo bằng thước)
# # Giả sử bàn làm việc rộng 300mm x 300mm
# pts_dst = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

# # 3. Tính Ma trận
# real_matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

# print("Đây là Ma trận chuẩn của bạn:")
# print("np.array(" + str(real_matrix.tolist()) + ")")

import cv2 as cv
import numpy as np

# --- CẤU HÌNH KÍCH THƯỚC THỰC TẾ (Bạn cần sửa dòng này) ---
# Đo khoảng cách hình chữ nhật trên bàn (đơn vị mm)
REAL_WIDTH_MM = 210   # Ví dụ: Chiều ngang 200mm
REAL_HEIGHT_MM = 297  # Ví dụ: Chiều dọc 150mm

# URL Camera của bạn
CAMERA_URL = "http://192.168.1.36:4747/video"

# --- KHÔNG SỬA DƯỚI DÒNG NÀY ---
clicked_points = []
window_name = "Calibration Tool - Click 4 Points"

def mouse_handler(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append((x, y))
            print(f"Đã click điểm {len(clicked_points)}: ({x}, {y})")

cv.namedWindow(window_name)
cv.setMouseCallback(window_name, mouse_handler)

cap = cv.VideoCapture(CAMERA_URL)

print("--- HƯỚNG DẪN ---")
print("1. Hãy click chuột vào 4 góc của hình chữ nhật trên màn hình.")
print("2. Thứ tự click BẮT BUỘC: Trái-Trên -> Phải-Trên -> Phải-Dưới -> Trái-Dưới")
print("   (Theo chiều kim đồng hồ, bắt đầu từ gốc 0,0)")
print("3. Nhấn 'r' để reset nếu click sai. Nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Vẽ các điểm đã click lên hình để dễ nhìn
    for i, pt in enumerate(clicked_points):
        cv.circle(frame, pt, 5, (0, 0, 255), -1) # Chấm đỏ
        cv.putText(frame, str(i+1), (pt[0]+10, pt[1]), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Vẽ khung chữ nhật nối các điểm để hình dung
        if i > 0:
            cv.line(frame, clicked_points[i-1], pt, (255, 0, 0), 2)
        if i == 3:
             cv.line(frame, clicked_points[3], clicked_points[0], (255, 0, 0), 2)

    cv.imshow(window_name, frame)
    key = cv.waitKey(1)

    # Nếu đã đủ 4 điểm, tính toán ngay
    if len(clicked_points) == 4:
        # Tọa độ pixel (Source)
        pts_src = np.float32(clicked_points)
        
        # Tọa độ thực tế (Destination) - Map theo thứ tự click
        # 1. Trái-Trên (0,0) -> 2. Phải-Trên (W,0) -> 3. Phải-Dưới (W,H) -> 4. Trái-Dưới (0,H)
        pts_dst = np.float32([
            [0, 0],                       
            [REAL_WIDTH_MM, 0],           
            [REAL_WIDTH_MM, REAL_HEIGHT_MM], 
            [0, REAL_HEIGHT_MM]           
        ])

        # Tính Ma trận Homography
        matrix = cv.getPerspectiveTransform(pts_src, pts_dst)
        
        print("\n" + "="*40)
        print("XONG! ĐÂY LÀ MA TRẬN CỦA BẠN (Copy dòng dưới):")
        print("="*40)
        
        # Định dạng output để bạn dễ copy
        np.set_printoptions(suppress=True, precision=6)
        matrix_str = str(matrix.tolist())
        print(f"REAL_MATRIX = np.array({matrix_str})")
        print("="*40 + "\n")
        
        # Dừng hình để bạn copy xong thì bấm q thoát
        cv.waitKey(0) 
        break

    if key & 0xFF == ord('q'):
        break
    if key & 0xFF == ord('r'): # Reset
        clicked_points = []
        print("Đã reset điểm click.")

cap.release()
cv.destroyAllWindows()