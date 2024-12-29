import os
import cv2
import numpy as np

# Đường dẫn thư mục chứa ảnh gốc và thư mục lưu kết quả
input_folder = r'C:\Users\acer\OneDrive - Hochiminh City University of Education\Documents\PythonProject-main\PythonProject-main\test_image'
output_folder = r'C:\Users\acer\OneDrive - Hochiminh City University of Education\Documents\PythonProject-main\PythonProject-main\Processed'
os.makedirs(output_folder, exist_ok=True)

def adjust_brightness(img, gamma=1.5):
    """
    Tăng sáng ảnh bằng cách áp dụng gamma correction.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def denoise_image(img):
    """
    Lọc nhiễu ảnh bằng Gaussian Blur.
    """
    return cv2.GaussianBlur(img, (5, 5), 0)

def process_images(input_folder, output_folder):
    """
    Tiền xử lý ảnh trong thư mục: tăng sáng và lọc nhiễu.
    """
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Đọc ảnh
            img = cv2.imread(image_path)
            if img is None:
                print(f"Không thể đọc ảnh: {image_path}")
                continue

            # Tăng sáng
            bright_img = adjust_brightness(img, gamma=1.5)

            # Lọc nhiễu
            processed_img = denoise_image(bright_img)

            # Lưu ảnh sau tiền xử lý
            cv2.imwrite(output_path, processed_img)
            print(f"Daxulyvaluuanh: {output_path}")

# Thực thi tiền xử lý
process_images(input_folder, output_folder)
