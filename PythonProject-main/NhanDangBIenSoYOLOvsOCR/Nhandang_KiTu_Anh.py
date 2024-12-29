import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import numpy as np
from pymongo import MongoClient
from datetime import datetime
import re

# Đường dẫn mô hình YOLOv8
yolo_model_path = r'C:\Users\acer\OneDrive - Hochiminh City University of Education\Documents\Tieuluan\PythonProject-main\modelsYOLO\best.pt'

# Kết nối MongoDB
mongodb_uri = "mongodb://localhost:27017"  # Thay bằng URI của bạn
client = MongoClient(mongodb_uri)
db = client["license_plate_db"]  # Tên cơ sở dữ liệu
collection = db["recognition_results"]  # Tên bộ sưu tập (collection)

# Khởi tạo YOLOv8 và PaddleOCR
yolo_model = YOLO(yolo_model_path)
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def advanced_preprocess(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    edges = cv2.Canny(img, 50, 150)  # Phát hiện biên cạnh
    return edges

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Chuyển về ảnh xám
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Giảm nhiễu
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Nhị phân hóa Otsu
    return binary

def enhance_sharpness(image):
    """
    Làm sắc nét ảnh bằng cách áp dụng bộ lọc.
    """
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Bộ lọc sắc nét
    sharp_img = cv2.filter2D(image, -1, kernel)
    return sharp_img

def detect_license_plate(img):
    """
    Phát hiện biển số trong ảnh bằng YOLOv8.
    """
    # Phát hiện đối tượng bằng YOLOv8
    results = yolo_model(img)

    # Lấy các vùng chứa biển số
    plates = []
    boxes = []
    if results[0].boxes:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Tọa độ bounding box
            plate_img = img[y1:y2, x1:x2]  # Cắt vùng biển số
            plates.append(plate_img)
            boxes.append((x1, y1, x2, y2))

    return plates, boxes


def recognize_text(plate_img):
    """
    Nhận dạng văn bản từ một ảnh biển số bằng PaddleOCR.
    """
    # Chuyển ảnh sang định dạng RGB (nếu cần)
    if len(plate_img.shape) == 2 or plate_img.shape[2] != 3:
        plate_img = cv2.cvtColor(plate_img, cv2.COLOR_GRAY2RGB)

    # Gọi PaddleOCR
    result = ocr.ocr(plate_img, cls=True)

    if not result or not result[0]:  # Kiểm tra nếu kết quả rỗng
        print("Không thể nhận dạng văn bản từ biển số.")
        return ""

    # Lấy text từ kết quả OCR
    texts = [line[1][0] for line in result[0]]

    return " ".join(texts)


def normalize_plate(plate_number):
    """
    Chuẩn hóa biển số xe: 
    - Xóa các ký tự đặc biệt như dấu gạch ngang (-), dấu chấm (.).
    - Chuyển đổi tất cả chữ cái sang in hoa.
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

# Bộ nhớ đệm tạm thời để tránh trùng trong phiên
recognized_cache = set()

def save_to_mongodb(image_path, plate_number, vehicle_type):
    """
    Lưu dữ liệu biển số vào MongoDB.
    """
    normalized_plate = normalize_plate(plate_number)

    if normalized_plate in recognized_cache:
        print(f"Biển số '{normalized_plate}' đã được nhận dạng trước đó.")
        return

    if not validate_vietnam_license_plate(normalized_plate):
        print(f"Biển số '{normalized_plate}' không đúng định dạng biển số xe Việt Nam.")
        return

    if collection.find_one({"plate_number": normalized_plate}):
        print(f"Biển số '{normalized_plate}' đã tồn tại trong cơ sở dữ liệu.")
        recognized_cache.add(normalized_plate)
        return

    data = {
        "image_path": image_path,
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


def main(image_path):
    """
    Nhận diện và lưu biển số từ ảnh.
    """
    # Đọc ảnh từ file
    img = cv2.imread(image_path)
    if img is None:
        print(f"Không thể đọc ảnh từ đường dẫn: {image_path}")
        return

    # Phát hiện biển số
    plates, boxes = detect_license_plate(img)

    if not plates:
        print("Không tìm thấy biển số nào.")
        return

    for i, (plate, (x1, y1, x2, y2)) in enumerate(zip(plates, boxes)):
        print(f"Đang xử lý biển số {i + 1}...")

        # Nhận diện văn bản
        recognized_text = recognize_text(plate)
        normalized_text = normalize_plate(recognized_text)
        vehicle_type = validate_vietnam_license_plate(normalized_text)

        if vehicle_type:
            print(f"Biển số hợp lệ ({vehicle_type}): {normalized_text}")
            # Lưu biển số vào MongoDB
            save_to_mongodb(image_path, normalized_text, vehicle_type)
        else:
            print(f"Biển số không hợp lệ: {recognized_text}")

        # Hiển thị ảnh với vùng chứa biển số
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, recognized_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Hiển thị ảnh sau khi xử lý
    cv2.imshow("KetQuaNhanDien", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Thay đường dẫn tới ảnh của bạn
    image_path = r'C:\Users\acer\OneDrive - Hochiminh City University of Education\Documents\Tieuluan\PythonProject-main\test_image\11.jpg'
    main(image_path)