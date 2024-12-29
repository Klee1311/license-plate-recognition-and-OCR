import cv2
import re
from ultralytics import YOLO
from paddleocr import PaddleOCR
from pymongo import MongoClient
from datetime import datetime

# Đường dẫn mô hình YOLOv8
yolo_model_path = r'C:\Users\acer\OneDrive - Hochiminh City University of Education\Documents\Tieuluan\PythonProject-main\modelsYOLO\best.pt'

# Kết nối MongoDB
mongodb_uri = "mongodb://localhost:27017"  # Thay bằng URI của bạn
client = MongoClient(mongodb_uri)
db = client["license_plate_db"]
collection = db["recognition_results"]

# Bộ nhớ đệm tạm thời để tránh trùng trong phiên
recognized_cache = set()

# Khởi tạo YOLOv8 và PaddleOCR
yolo_model = YOLO(yolo_model_path)
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def normalize_plate(plate_number):
    """
    Chuẩn hóa biển số xe: xóa khoảng trắng, gạch ngang và chuyển về dạng chuẩn.
    """
    return re.sub(r'[-\s\.]', '', plate_number).upper()

def validate_vietnam_license_plate(plate_number):
    """
    Xác thực định dạng biển số xe Việt Nam:
    - Ô tô: 2 chữ số + 1 chữ cái + 3 số + 2 số
    - Xe máy: 
      + Trường hợp 1: 2 chữ số + 2 chữ cái + 3 số + 2 số
      + Trường hợp 2: 2 chữ số + 1 chữ cái + 1 chữ số + 3 số + 2 số
    """
    car_pattern = r"^\d{2}[A-Z]\d{3}\d{2}$"  # Định dạng ô tô
    bike_pattern_1 = r"^\d{2}[A-Z]{2}\d{3}\d{2}$"  # Trường hợp 1: 2 chữ cái
    bike_pattern_2 = r"^\d{2}[A-Z]\d\d{3}\d{2}$"  # Trường hợp 2: 1 chữ cái + 1 số

    if re.match(car_pattern, plate_number):
        return "Ô tô"
    elif re.match(bike_pattern_1, plate_number):
        return "Xe máy (Trường hợp 1: 2 chữ cái)"
    elif re.match(bike_pattern_2, plate_number):
        return "Xe máy (Trường hợp 2: 1 chữ cái + 1 số)"
    return None

# Ngưỡng độ tin cậy cho mô hình YOLOv8 
confidence_threshold = 0.85

def detect_license_plate(frame):
    """
    Phát hiện biển số trong khung hình bằng YOLOv8.
    Lọc kết quả dựa trên độ tin cậy.
    """
    results = yolo_model(frame)
    plates, boxes = [], []
    if results[0].boxes:
        for box in results[0].boxes:
            confidence = box.conf[0]  # Lấy độ tin cậy của dự đoán
            if confidence >= confidence_threshold:  # Chỉ giữ lại dự đoán có độ tin cậy cao hơn ngưỡng
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_img = frame[y1:y2, x1:x2]
                plates.append(plate_img)
                boxes.append((x1, y1, x2, y2))
    return plates, boxes

def recognize_text(plate_img):
    """
    Nhận dạng văn bản từ một ảnh biển số bằng PaddleOCR.
    """
    if len(plate_img.shape) == 2 or plate_img.shape[2] != 3:
        plate_img = cv2.cvtColor(plate_img, cv2.COLOR_GRAY2RGB)
    result = ocr.ocr(plate_img, cls=True)
    texts = [line[1][0] for line in result[0]] if result and result[0] else []
    return " ".join(texts) if texts else ""

def save_to_mongodb(plate_number):
    """
    Kiểm tra và lưu biển số hợp lệ vào MongoDB nếu chưa tồn tại.
    """
    normalized_plate = normalize_plate(plate_number)

    if normalized_plate in recognized_cache:
        print(f"Biển số '{normalized_plate}' đã được nhận dạng trước đó.")
        return

    vehicle_type = validate_vietnam_license_plate(normalized_plate)
    if not vehicle_type:
        print(f"Biển số '{normalized_plate}' không đúng định dạng biển số xe Việt Nam.")
        return

    if collection.find_one({"plate_number": normalized_plate}):
        print(f"Biển số '{normalized_plate}' đã tồn tại trong cơ sở dữ liệu.")
        recognized_cache.add(normalized_plate)
        return

    data = {
        "plate_number": normalized_plate,
        "vehicle_type": vehicle_type,
        "recognized_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    try:
        result = collection.insert_one(data)
        recognized_cache.add(normalized_plate)
        print(f"Đã lưu biển số {normalized_plate} ({vehicle_type}) vào MongoDB. ID: {result.inserted_id}")
    except Exception as e:
        print(f"Lỗi khi lưu vào MongoDB: {e}")

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở webcam.")
        return

    print("Nhấn 'q' để thoát.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc khung hình từ webcam.")
            break

        plates, boxes = detect_license_plate(frame)
        for plate, (x1, y1, x2, y2) in zip(plates, boxes):
            # Nhận diện và chuẩn hóa biển số
            recognized_text = recognize_text(plate)
            normalized_text = normalize_plate(recognized_text)

            if normalized_text:
                # Xác thực định dạng biển số
                vehicle_type = validate_vietnam_license_plate(normalized_text)
                if vehicle_type:
                    save_to_mongodb(normalized_text)
                    color = (0, 255, 0)  # Màu xanh lá cho biển số hợp lệ
                else:
                    color = (0, 0, 255)  # Màu đỏ cho biển số không hợp lệ

                # Vẽ khung và hiển thị biển số
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, normalized_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Nhandienbiensoxe", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
