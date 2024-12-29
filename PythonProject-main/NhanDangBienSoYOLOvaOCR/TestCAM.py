from flask import Flask, request, jsonify, send_from_directory, Response
import cv2
import threading
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR

app = Flask(__name__)

# Đường dẫn mô hình YOLOv8
yolo_model_path = r'C:\Users\acer\OneDrive - Hochiminh City University of Education\Documents\PythonProject-main\PythonProject-main\modelsYOLO\best.pt'

# Khởi tạo YOLOv8 và PaddleOCR
yolo_model = YOLO(yolo_model_path)
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Biến toàn cục
last_recognized_text = "Không nhận dạng được"
is_running = False
frame_to_display = None  # Khung hình hiện tại


def detect_license_plate(frame):
    results = yolo_model(frame)
    if results[0].boxes is None or len(results[0].boxes) == 0:
        return [], []

    plates, boxes = [], []
    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        plate_img = frame[y1:y2, x1:x2]
        plates.append(plate_img)
        boxes.append((x1, y1, x2, y2))

    return plates, boxes


def recognize_text(plate_img):
    if len(plate_img.shape) == 2 or plate_img.shape[2] != 3:
        plate_img = cv2.cvtColor(plate_img, cv2.COLOR_GRAY2RGB)

    result = ocr.ocr(plate_img, cls=True)
    if not result or not result[0]:
        return "Không nhận dạng được"

    texts = [line[1][0] for line in result[0]]
    return " ".join(texts)


def capture_video():
    global last_recognized_text, is_running, frame_to_display
    cap = cv2.VideoCapture(0)

    while is_running:
        ret, frame = cap.read()
        if not ret:
            continue

        plates, boxes = detect_license_plate(frame)
        for plate, (x1, y1, x2, y2) in zip(plates, boxes):
            recognized_text = recognize_text(plate)
            if recognized_text:
                last_recognized_text = recognized_text

            # Vẽ bounding box và ký tự nhận dạng
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, recognized_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        frame_to_display = frame

    cap.release()


@app.route("/")
def index():
    return send_from_directory('static', 'index.html')


@app.route('/video_feed')
def video_feed():
    def generate():
        global frame_to_display
        while is_running:
            if frame_to_display is not None:
                _, buffer = cv2.imencode('.jpg', frame_to_display)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                # Nếu không có frame, đợi một chút trước khi thử lại
                cv2.waitKey(1)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/detect', methods=['POST'])
def start_detection():
    global is_running, capture_thread
    if is_running:
        return jsonify({"status": "Already running"})

    is_running = True
    capture_thread = threading.Thread(target=capture_video)
    capture_thread.start()
    return jsonify({"status": "Started"})


@app.route('/stop', methods=['POST'])
def stop_detection():
    global is_running
    is_running = False
    return jsonify({"recognized_text": last_recognized_text})


if __name__ == "__main__":
    app.run(debug=True)
