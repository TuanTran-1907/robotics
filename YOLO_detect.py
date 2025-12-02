import cv2 as cv
from ultralytics import YOLO

model = YOLO('yolo11s.onnx', task='detect')

camera = cv.VideoCapture("http://192.168.1.36:4747/video")

while True:
    ret,frame = camera.read()

    if not ret:
        break
    results = model(frame,verbose = False, imgsz=1280, conf=0.5)

    plot_frame = results[0].plot()

    cv.imshow("Yolo", plot_frame)

    if cv.waitKey(1) and 0xFF == ord('q'):
        break

camera.release()