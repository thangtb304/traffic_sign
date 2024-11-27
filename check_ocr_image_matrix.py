import cv2
import pytesseract
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt

# Đặt đường dẫn tới tesseract (nếu cần thiết, đặc biệt trên Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Đường dẫn tùy thuộc vào nơi bạn cài đặt

# Hàm nhận diện biển số xe và hiển thị độ tin cậy
def recognize_license_plate_mt(image_matrix):
 
    # 5. Áp dụng OCR để nhận diện biển số xe và lấy độ tin cậy
    custom_config = r'--oem 3 --psm 6'  # Tùy chọn cấu hình cho OCR (tùy chỉnh nếu cần thiết)
    
    # Sử dụng image_to_data để lấy thông tin chi tiết về độ tin cậy
    data = pytesseract.image_to_data(image_matrix, config=custom_config, output_type=pytesseract.Output.DICT)

    # 6. Lấy ra văn bản nhận diện và độ tin cậy
    text = ''
    lap=0
    list_text=[]
    total_confidence = 0
    num_words = 0
    max_confidence = 0  # Khởi tạo biến lưu độ tin cậy cao nhất
    min_confidence = 100  # Khởi tạo biến lưu độ tin cậy thấp nhất
    for i in range(len(data['text'])):
        lap+=1
        word = data['text'][i]
        confidence = int(data['conf'][i]) if data['conf'][i] != '-1' else 0  # Độ tin cậy (nếu có)
        #print(confidence)
        if word.strip():  # Kiểm tra nếu từ có nội dung
            text += word + ' '
            list_text.append(word)
            total_confidence += confidence
            num_words += 1
            if confidence > max_confidence:
                max_confidence = confidence  # Cập nhật độ tin cậy cao nhất
            if confidence < min_confidence:
                min_confidence = confidence  # Cập nhật độ tin cậy cao nhất
        ######

    
    # Tính toán độ tin cậy trung bình
    average_confidence = total_confidence / num_words if num_words > 0 else 0

    # 7. Trả về kết quả nhận diện, độ tin cậy trung bình và độ tin cậy cao nhất
    return text.strip(), average_confidence, max_confidence, min_confidence, list_text,lap

import easyocr    #easyocr quá nặng
def recognize_lp_easyocr(image_matrix):
     # Khởi tạo đối tượng EasyOCR
    reader = easyocr.Reader(['en'])  # 'en' là ngôn ngữ nhận diện (có thể thêm 'vi' nếu cần)

    # 5. Áp dụng OCR để nhận diện biển số xe
    results = reader.readtext(image_matrix)

    # 6. Lấy ra văn bản nhận diện và độ tin cậy
    text = ''
    lap = 0
    list_text = []
    total_confidence = 0
    num_words = 0
    max_confidence = 0  # Khởi tạo biến lưu độ tin cậy cao nhất
    min_confidence = 100  # Khởi tạo biến lưu độ tin cậy thấp nhất

    for (bbox, word, confidence) in results:
        lap += 1
        word = word.strip()  # Loại bỏ khoảng trắng ở đầu/cuối
        if word:
            text += word + ' '
            list_text.append(word)
            total_confidence += confidence
            num_words += 1
            if confidence > max_confidence:
                max_confidence = confidence  # Cập nhật độ tin cậy cao nhất
            if confidence < min_confidence:
                min_confidence = confidence  # Cập nhật độ tin cậy thấp nhất

    # Tính toán độ tin cậy trung bình
    average_confidence = total_confidence / num_words if num_words > 0 else 0

    # 7. Trả về kết quả nhận diện, độ tin cậy trung bình và độ tin cậy cao nhất
    return text.strip(), average_confidence, max_confidence, min_confidence, list_text, lap


###3333

from paddleocr import PaddleOCR
import cv2


def recognize_lp_paddleocr(image_matrix, padding=10):
    """
    Nhận diện biển số xe với PaddleOCR và sử dụng padding để tập trung vào vùng biển số.
    
    Args:
        image_matrix (numpy.ndarray): Ma trận ảnh cần xử lý.
        padding (int): Kích thước padding thêm vào vùng nhận diện (pixels).
    
    Returns:
        tuple: Gồm (text, average_confidence, max_confidence, min_confidence, list_text, lap)
    """
    # Khởi tạo đối tượng PaddleOCR
    ocr = PaddleOCR(use_angle_cls=False, use_gpu=True, lang='en', show_log=False)

    # Kiểm tra xem ảnh có hợp lệ không
    # if image_matrix is None:
    #     print("Warning: image_matrix is None")
    #     return '', 0, 0, 100, [], 0
    """
    '' (chuỗi rỗng): Không nhận dạng được text biển số
    0: Độ tin cậy trung bình (average_confidence)
    0: Độ tin cậy tối đa (max_confidence)
    100: Độ tin cậy tối thiểu (min_confidence)
    []: Danh sách các từ nhận dạng được (list_text)
    0: Số lượng từ được nhận dạng (lap)
    """

    try:
        # Áp dụng OCR
        results = ocr.ocr(image_matrix)
        
        # Lấy kết quả nhận diện
        text = ''
        lap = 0
        list_text = []
        total_confidence = 0
        num_words = 0
        max_confidence = 0
        min_confidence = 100


        for result in results[0]:  # results[0] chứa các kết quả OCR
            bbox, word_info = result[0], result[1]
            word, confidence = word_info[0], word_info[1]
            lap += 1
            word = word.strip()
            if word:
                text += word + ' '
                list_text.append(word)
                total_confidence += confidence
                num_words += 1
                max_confidence = max(max_confidence, confidence)
                min_confidence = min(min_confidence, confidence)

        # Tính độ tin cậy trung bình
        average_confidence = total_confidence / num_words if num_words > 0 else 0

        return text.strip(), average_confidence, max_confidence, min_confidence, list_text, lap

    except Exception as e:
        print(f"Error in OCR recognition: {e}")
        return '', 0, 0, 100, [], 0