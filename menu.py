import customtkinter as ctk
import cv2
import time  # <--- THÊM DÒNG NÀY
from ultralytics import YOLO
from PIL import Image, ImageTk
import threading
# =============================
# CẤU HÌNH HỆ THỐNG
# =============================
MODEL_PATH = "best (1).onnx"  # Đảm bảo file này đúng tên và vị trí
IMG_SIZE = 640
CONF_THRES = 0.5
IOU_THRES = 0.5

MENU_ITEMS = [
    "coca", "pepsi", "gio_banh", "gio_coca", "gio_pepsi", "gio_sprite",
    "gio_fanta", "pho_tron", "xua_nay", "siu_cay", "omachi", "hao_hao",
    "cay_mai", "cay_quat", "nuoc", "lon_coca", "lon_pepsi", "ket_coca"
]

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Robot Vision - Chọn Món & Xác Nhận")
        self.geometry("1280x720")
        
        # --- QUẢN LÝ TRẠNG THÁI ---
        self.selected_classes = []   # Danh sách đang chọn (chưa detect)
        self.confirmed_classes = []  # Danh sách đã xác nhận (đang detect)
        self.running = True

        # Thêm vào trong __init__ của class App
        self.picking_queue = []  # Danh sách các vật cần gắp
        self.is_robot_busy = False  # Trạng thái: Robot đang bận gắp hay rảnh?
        self.last_sent_time = 0     # Để delay nếu không có tín hiệu phản hồi thật

        # --- LOAD MODEL ---
        print("--- Đang tải model YOLO ---")
        try:
            self.model = YOLO(MODEL_PATH, task='detect')
            self.class_names = self.model.names
            print("Model đã sẵn sàng!")
        except Exception as e:
            print(f"Lỗi load model: {e}")
            self.model = None

        # --- GIAO DIỆN ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)

        # 1. Menu bên trái
        self.frame_menu = ctk.CTkFrame(self)
        self.frame_menu.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        ctk.CTkLabel(self.frame_menu, text="DANH SÁCH MÓN", font=("Arial", 20, "bold")).pack(pady=10)
        
        # Vùng chứa nút cuộn được
        self.scroll_frame = ctk.CTkScrollableFrame(self.frame_menu, label_text="Chọn món")
        self.scroll_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.buttons = {}
        for item in MENU_ITEMS:
            btn = ctk.CTkButton(
                self.scroll_frame, 
                text=item, 
                fg_color="transparent", 
                border_width=2,
                command=lambda x=item: self.toggle_selection(x)
            )
            btn.pack(pady=3, fill="x")
            self.buttons[item] = btn

        # --- CÁC NÚT ĐIỀU KHIỂN ---
        self.lbl_info = ctk.CTkLabel(self.frame_menu, text="Trạng thái: CHỜ CHỌN MÓN", text_color="orange")
        self.lbl_info.pack(pady=5)

        # Nút XÁC NHẬN (Quan trọng nhất)
        self.btn_confirm = ctk.CTkButton(
            self.frame_menu, 
            text="XÁC NHẬN (Bắt đầu)", 
            fg_color="green", 
            height=40,
            font=("Arial", 14, "bold"),
            command=self.confirm_selection
        )
        self.btn_confirm.pack(pady=10, fill="x", padx=10)

        # Nút CLEAR
        self.btn_clear = ctk.CTkButton(
            self.frame_menu, 
            text="HỦY / CHỌN LẠI", 
            fg_color="red", 
            command=self.clear_selection
        )
        self.btn_clear.pack(pady=5, fill="x", padx=10)

        # 2. Màn hình Camera bên phải
        self.frame_cam = ctk.CTkFrame(self)
        self.frame_cam.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        self.lbl_video = ctk.CTkLabel(self.frame_cam, text="Đang tìm camera...")
        self.lbl_video.pack(expand=True, fill="both")

        # --- KHỞI ĐỘNG CAMERA THÔNG MINH ---
        self.cap = None
        self.start_camera()

    def start_camera(self):
        """Thử tìm camera ở các cổng khác nhau"""
        for i in range(2): # Thử cổng 0 và 1
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(i)
            
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    self.cap = cap
                    self.cap.set(3, 1280) # Width
                    self.cap.set(4, 720)  # Height
                    print(f"Đã kết nối Camera index {i}")
                    threading.Thread(target=self.video_loop, daemon=True).start()
                    return
        
        self.lbl_video.configure(text="LỖI: KHÔNG TÌM THẤY CAMERA")

    # --- XỬ LÝ SỰ KIỆN ---
    def toggle_selection(self, item_name):
        """Chọn món nhưng chưa detect ngay"""
        if item_name in self.selected_classes:
            self.selected_classes.remove(item_name)
            self.buttons[item_name].configure(fg_color="transparent")
        else:
            self.selected_classes.append(item_name)
            self.buttons[item_name].configure(fg_color="#1f6aa5") # Màu xanh dương
        
        self.lbl_info.configure(text=f"Đã chọn tạm: {len(self.selected_classes)} món")

    def confirm_selection(self):
        """Bấm nút này mới bắt đầu Detect"""
        if not self.selected_classes:
            self.lbl_info.configure(text="Lỗi: Chưa chọn món nào!", text_color="red")
            return
        
        # Sao chép danh sách chọn sang danh sách detect
        self.confirmed_classes = self.selected_classes.copy()
        
        ds_mon = ", ".join(self.confirmed_classes)
        self.lbl_info.configure(text=f"ĐANG TÌM: {ds_mon}", text_color="#00FF00")
        print(f"CONFIRMED: Bắt đầu tìm {self.confirmed_classes}")
        
    def clear_selection(self):
        """Dừng detect, xóa chọn và xóa hàng đợi"""
        self.selected_classes.clear()
        self.confirmed_classes.clear()
        self.picking_queue.clear()     # <--- THÊM: Xóa hàng đợi
        self.is_robot_busy = False     # <--- THÊM: Reset trạng thái robot
        
        for btn in self.buttons.values():
            btn.configure(fg_color="transparent")
            
        self.lbl_info.configure(text="Đã hủy. Mời chọn lại.", text_color="orange")

    def bbox_center(self, x1, y1, x2, y2):
        return int((x1+x2)/2), int((y1+y2)/2)

    # --- LUỒNG XỬ LÝ HÌNH ẢNH ---
    # def video_loop(self):
    #     while self.running:
    #         ret, frame = self.cap.read()
    #         if not ret: continue

    #         # 1. GIỮ NGUYÊN FRAME GỐC ĐỂ DETECT (Giống code gốc của bạn)
    #         # Không resize, không đổi màu frame này trước khi đưa vào YOLO
    #         results = None
    #         if self.model and self.confirmed_classes:
    #             results = self.model(
    #                 frame,             # Dùng frame gốc 1280x720
    #                 imgsz=IMG_SIZE, 
    #                 conf=CONF_THRES, 
    #                 iou=IOU_THRES, 
    #                 verbose=False
    #             )

    #         # 2. XỬ LÝ VẼ (Vẽ lên frame gốc để đảm bảo tọa độ chính xác)
    #         detections = []
    #         if results and results[0].boxes:
    #             for box in results[0].boxes:
    #                 cls_id = int(box.cls[0])
    #                 cls_name = self.class_names[cls_id]
    #                 if cls_name in self.confirmed_classes:
    #                     conf = float(box.conf[0])
    #                     x1, y1, x2, y2 = map(int, box.xyxy[0])
    #                     detections.append({"class": cls_name, "conf": conf, "bbox": (x1, y1, x2, y2)})
                        
    #                     # Vẽ như code gốc của bạn
    #                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #                     cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 10),
    #                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    #         # Tìm vật tốt nhất để gửi Robot (Giống code gốc)
    #         if detections:
    #             best = max(detections, key=lambda x: x["conf"])
    #             cx, cy = self.bbox_center(*best["bbox"])
    #             print(f"{best['class']},{cx},{cy}") # Gửi robot
    #             cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    #         # 3. HIỂN THỊ (Chỉ chuyển đổi màu tại bước cuối cùng)
    #         # Chuyển từ BGR sang RGB
    #         rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         pil_img = Image.fromarray(rgb_image)
            
    #         # Sử dụng CTkImage để tự động xử lý scaling mượt mà
    #         ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(900, 500))
            
    #         self.lbl_video.configure(image=ctk_img)
    #         self.lbl_video.image = ctk_img
    def video_loop(self):
        """Luồng xử lý hình ảnh và logic điều khiển Robot theo hàng đợi"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret: continue

            # --- PHẦN 1: LOGIC TÌM KIẾM & HÀNG ĐỢI (Queue) ---
            
            # A. GIAI ĐOẠN QUÉT (Chỉ quét khi chưa có hàng đợi và Robot đang rảnh)
            # Nghĩa là: Làm xong hết đợt cũ mới được tìm đợt mới.
            if not self.picking_queue and not self.is_robot_busy:
                detections = []
                # Chỉ chạy model khi đã bấm nút XÁC NHẬN
                if self.model and self.confirmed_classes:
                    # Chạy YOLO
                    results = self.model(frame, imgsz=IMG_SIZE, conf=CONF_THRES, verbose=False)
                    
                    if results[0].boxes:
                        for box in results[0].boxes:
                            cls_name = self.class_names[int(box.cls[0])]
                            # Chỉ lấy những món nằm trong danh sách xác nhận
                            if cls_name in self.confirmed_classes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                cx, cy = self.bbox_center(x1, y1, x2, y2)
                                detections.append({
                                    "class": cls_name, 
                                    "cx": cx, 
                                    "cy": cy,
                                    "bbox": (x1, y1, x2, y2) 
                                })

                # Nếu tìm thấy vật, nạp vào hàng đợi và SẮP XẾP
                if detections:
                    # Sắp xếp từ Trái sang Phải (cx nhỏ -> cx lớn)
                    detections.sort(key=lambda k: k['cx']) 
                    
                    self.picking_queue = detections # Lưu vào hàng đợi
                    print(f"--- Đã tìm thấy {len(self.picking_queue)} vật. Bắt đầu xếp hàng gắp ---")

            # B. GIAI ĐOẠN GỬI TÍN HIỆU (Xử lý từng món trong hàng đợi)
            current_time = time.time()
            
            # Điều kiện gửi: Có hàng VÀ (Robot đang rảnh HOẶC đã quá 5 giây mà chưa thấy xong)
            if self.picking_queue and (not self.is_robot_busy or (current_time - self.last_sent_time > 5)):
                
                # Lấy vật ĐẦU TIÊN ra khỏi hàng đợi (pop(0))
                target = self.picking_queue.pop(0) 
                
                # Tạo bản tin gửi Robot (Ví dụ: "coca,320,240")
                msg = f"{target['class']},{target['cx']},{target['cy']}"
                print(f">>> GỬI ROBOT: {msg} (Còn lại trong hàng: {len(self.picking_queue)})")
                
                # --- CHỖ NÀY ĐỂ CODE GỬI SERIAL SAU NÀY ---
                # if self.ser: self.ser.write((msg + '\n').encode())

                # Đánh dấu Robot đang bận để vòng lặp sau không gửi tiếp
                self.is_robot_busy = True
                self.last_sent_time = current_time

            # --- PHẦN 2: VẼ GIAO DIỆN (Visual) ---
            
            # Vẽ các vật ĐANG CHỜ trong hàng đợi (để người dùng biết còn bao nhiêu cái)
            for item in self.picking_queue:
                x1, y1, x2, y2 = item['bbox']
                cx, cy = item['cx'], item['cy']
                
                # Vẽ khung màu xanh dương (biểu thị trạng thái CHỜ)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
                cv2.putText(frame, f"WAIT: {item['class']}", (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            # Vẽ thông báo trạng thái chung lên góc màn hình
            status_text = f"Queue: {len(self.picking_queue)} | Robot: {'BUSY' if self.is_robot_busy else 'READY'}"
            cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # --- PHẦN 3: HIỂN THỊ LÊN APP ---
            # Chuyển đổi màu và hiển thị (Giữ nguyên logic hiển thị cũ của bạn)
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_image)
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(900, 500))
            
            self.lbl_video.configure(image=ctk_img)
            self.lbl_video.image = ctk_img

    def on_close(self):
        self.running = False
        if self.cap: self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()