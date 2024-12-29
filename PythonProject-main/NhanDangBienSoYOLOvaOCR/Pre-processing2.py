import os
import cv2
import numpy as np

# Đường dẫn thư mục chứa ảnh gốc và thư mục lưu ảnh sau tiền xử lý
input_folder = r'C:\Users\acer\OneDrive - Hochiminh City University of Education\Documents\PythonProject-main\PythonProject-main\test_image'
output_folder = r'C:\Users\acer\OneDrive - Hochiminh City University of Education\Documents\PythonProject-main\PythonProject-main\Processed2'
os.makedirs(output_folder, exist_ok=True)

def enhance_image(image):
    """
    Tiền xử lý ảnh bao gồm tăng sáng, tăng độ tương phản, lọc nhiễu, và tăng độ nét.
    """
    # Tăng sáng bằng cách nhân hệ số
    brightness_factor = 1.5  # Hệ số tăng sáng
    brightened_image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)

    # Tăng độ tương phản bằng CLAHE
    lab = cv2.cvtColor(brightened_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    contrasted_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Lọc nhiễu bằng bộ lọc Gaussian
    denoised_image = cv2.GaussianBlur(contrasted_image, (5, 5), 0)

    # Tăng độ nét bằng cách cộng ảnh gốc với Laplacian
    sharp = cv2.Laplacian(denoised_image, cv2.CV_64F)
    sharp = np.uint8(np.clip(sharp, 0, 255))
    sharpened_image = cv2.addWeighted(denoised_image, 1.5, sharp, -0.5, 0)

    return sharpened_image

def process_images(input_folder):
    """
    Duyệt qua các file ảnh trong thư mục, thực hiện tiền xử lý và lưu ảnh.
    """
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Đọc ảnh
            image = cv2.imread(input_path)
            if image is None:
                print(f"Không thể đọc ảnh: {input_path}")
                continue

            # Tiền xử lý ảnh
            processed_image = enhance_image(image)

            # Lưu ảnh đã xử lý
            cv2.imwrite(output_path, processed_image)
            print(f"Daxulyvaluuanh: {output_path}")

# Thực hiện xử lý
process_images(input_folder)
