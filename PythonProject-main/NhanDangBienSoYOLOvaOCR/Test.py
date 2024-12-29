import os
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Đường dẫn model YOLO và thư mục ảnh
yolo_model_path = r'C:\Users\acer\OneDrive - Hochiminh City University of Education\Documents\PythonProject-main\PythonProject-main\modelsYOLO\best.pt'
image_folder = r'C:\Users\acer\OneDrive - Hochiminh City University of Education\Documents\PythonProject-main\PythonProject-main\test_image'

# Khởi tạo YOLO và PaddleOCR
yolo_model = YOLO(yolo_model_path)
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Khởi tạo OCR

# Tạo thư mục kết quả nếu chưa có
output_folder = r'C:\Users\acer\OneDrive - Hochiminh City University of Education\Documents\PythonProject-main\PythonProject-main\Results_test'
os.makedirs(output_folder, exist_ok=True)


def process_images(image_folder):
    """
    Duyệt qua các ảnh trong thư mục và xử lý từng ảnh.
    """
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            process_single_image(image_path)


def process_single_image(image_path):
    """
    Xử lý một ảnh duy nhất: phát hiện biển số, nhận diện ký tự và lưu kết quả.
    """
    # Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return

    # Phát hiện biển số xe bằng YOLOv8
    results = yolo_model(img)
    if not results or not results[0].boxes:  # Kiểm tra nếu không có kết quả
        print(f"Không phát hiện được biển số trong ảnh: {image_path}")
        with open(os.path.join(output_folder, "results.txt"), "a", encoding="utf-8") as f:
            f.write(f"{os.path.basename(image_path)} - Không nhận dạng được\n")
        return

    detections = results[0].boxes.xyxy.numpy()  # Lấy toạ độ bounding box
    if detections is None or len(detections) == 0:  # Kiểm tra nếu không có bounding box
        print(f"Không có bounding box trong ảnh: {image_path}")
        with open(os.path.join(output_folder, "results.txt"), "a", encoding="utf-8") as f:
            f.write(f"{os.path.basename(image_path)} - Không nhận dạng được\n")
        return

    detected_texts = []
    for x1, y1, x2, y2 in detections:
        # Cắt vùng chứa biển số
        cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
        if cropped_img.size == 0:
            print(f"Không thể cắt được ảnh tại {image_path}")
            continue

        # Nhận dạng ký tự bằng PaddleOCR
        cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        ocr_results = ocr.ocr(cropped_img_rgb, cls=True)

        # Kiểm tra tính hợp lệ của ocr_results
        if not ocr_results or not ocr_results[0]:
            print(f"OCR không nhận dạng được vùng: {image_path}")
            continue

        # Trích xuất văn bản từ kết quả OCR
        detected_text = " ".join([line[1][0] for line in ocr_results[0]])
        detected_texts.append(detected_text)

        # Vẽ bounding box và kết quả OCR lên ảnh gốc
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Vẽ khung xanh
        cv2.putText(img, detected_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2, cv2.LINE_AA)  # Thêm văn bản nhận diện

    # Kiểm tra nếu không có văn bản được nhận dạng
    if not detected_texts:
        print(f"Không nhận dạng được biển số trong ảnh: {image_path}")
        with open(os.path.join(output_folder, "results.txt"), "a", encoding="utf-8") as f:
            f.write(f"{os.path.basename(image_path)} - Không nhận dạng được\n")
    else:
        detected_text_combined = " | ".join(detected_texts)
        print(f"Kết quả nhận dạng từ {image_path}: {detected_text_combined}")

        # Lưu kết quả nhận dạng vào file
        with open(os.path.join(output_folder, "results.txt"), "a", encoding="utf-8") as f:
            f.write(f"{os.path.basename(image_path)} - {detected_text_combined}\n")

        # Lưu ảnh gốc có vẽ kết quả
        output_image_path = os.path.join(output_folder,
                                         f"{os.path.splitext(os.path.basename(image_path))[0]}_result.jpg")
        cv2.imwrite(output_image_path, img)


# Chạy xử lý
process_images(image_folder)
