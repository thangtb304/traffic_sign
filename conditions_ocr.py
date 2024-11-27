import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import pytesseract
from scipy.ndimage import variance
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class PlateQualityChecker:
    def __init__(self):
        self.min_plate_area = 10000  # Diện tích tối thiểu của biển số
        self.blur_threshold = 500  # Ngưỡng độ nét
        self.brightness_threshold = 60  # Ngưỡng độ sáng
        self.contrast_threshold = 60  # Ngưỡng độ tương phản
        
    def check_plate_size(self, plate_img):
        """Kiểm tra kích thước biển số có đủ lớn không"""
        height, width = plate_img.shape[:2]
        area = height * width
        return area >= self.min_plate_area
        #return area,height,width
    
    def check_blur(self, plate_img):
        """Kiểm tra độ nét của ảnh sử dụng Laplacian variance"""
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return blur_score > self.blur_threshold
        #return int(blur_score)
    
    def check_brightness_contrast(self, plate_img):
        """Kiểm tra độ sáng và độ tương phản"""
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        contrast = np.std(gray)
        return brightness > self.brightness_threshold and contrast > self.contrast_threshold
        #return int(brightness) , int(contrast)
    
    def check_plate_angle(self, plate_img):
        """Kiểm tra góc nghiêng của biển số"""
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 50)
        
        if lines is not None:
            for rho, theta in lines[0]:
                angle = theta * 180 / np.pi
                # Nếu góc nghiêng quá 15 độ, coi như biển số bị nghiêng
                return abs(90 - angle) < 35 or abs(angle) < 35
        return True
        
    def is_plate_quality_good(self, plate_img):
        """Kiểm tra tổng thể chất lượng ảnh biển số"""
        if not self.check_plate_size(plate_img):
            return False
            
        if not self.check_blur(plate_img):
            return False
            
        if not self.check_brightness_contrast(plate_img):
            return False
            
        if not self.check_plate_angle(plate_img):
            return False
            
        return True
    
    def area_roi(frame):
        # Xác định vùng ROI (bạn có thể thay đổi giá trị này)
        roi_x1, roi_y1, roi_x2, roi_y2 = 100, 100, 500, 300  # Các giá trị tùy chỉnh vùng ROI
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
