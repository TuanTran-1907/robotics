import cv2
from ultralytics import YOLO

# =============================
# 1. CONFIG
# =============================
MODEL_PATH = "best.pt"
IMG_SIZE = 640
CONF_THRES = 0.5
IOU_THRES = 0.5

# Các class bạn CẦN
NEEDED_CLASSES = []

mon = input("Hay chon mon muon lay: ")

selected_classes = [x.strip() for x in mon.split(",")]

NEEDED_CLASSES.extend(selected_classes)
# =============================
# 2. LOAD MODEL
# =============================
model = YOLO('best (1).onnx',task='detect')
class_names = model.names

# =============================
# 3. OPEN WEBCAM
# =============================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Set độ phân giải cho Rapoo C260
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("❌ Không mở được webcam")
    exit()

print("✅ Webcam & model đã sẵn sàng")

def bbox_center(x1, y1, x2, y2):
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cx, cy

# =============================
# 4. REAL-TIME LOOP
# =============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO inference
    results = model(
        frame,
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        iou=IOU_THRES,
        verbose=False
    )

    detections = []

    # =============================
    # 5. FILTER CLASS
    # =============================
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = class_names[cls_id]
        conf = float(box.conf[0])

        if cls_name not in NEEDED_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        detections.append({
            "class": cls_name,
            "conf": conf,
            "bbox": (x1, y1, x2, y2)
        })

    # =============================
    # 6. VẼ KẾT QUẢ
    # =============================
    if detections:
        best = max(detections, key=lambda x: x["conf"])

        x1, y1, x2, y2 = best["bbox"]
        cx, cy = bbox_center(x1, y1, x2, y2)

        # Chuỗi gửi robot
        msg = f"{best['class']},{cx},{cy}\n"
        # ser.write(msg.encode())
        print(msg)

        # DEBUG
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"({cx},{cy})", (cx+5, cy-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f'{det["class"]} {det["conf"]:.2f}'

        cv2.rectangle(frame, (x1, y1), (x2, y2),
                      (0, 255, 0), 2)

        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)


    # Hiển thị FPS
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cv2.putText(frame, f'FPS: {fps}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 255), 2)

    cv2.imshow("YOLO Robot Detect", frame)

    # ESC để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =============================
# 7. CLEANUP
# =============================
cap.release()
cv2.destroyAllWindows()
