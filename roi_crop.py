import cv2
import numpy as np
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from datetime import datetime
import uuid
# Hàm kiểm tra xem một điểm có nằm trong một đa giác không
def is_point_in_polygon(point, poly_vertices):
    """
    Kết quả trả về là một giá trị kiểu float, có ý nghĩa như sau:
    >= 0: Điểm nằm trong đa giác hoặc trên cạnh của đa giác.
    < 0: Điểm nằm ngoài đa giác.
    0: Điểm nằm trên một cạnh của đa giác.
    """
    # Kiểm tra nếu poly_vertices là list, nếu đúng thì chuyển thành NumPy array
    if not isinstance(poly_vertices, np.ndarray):
        poly_vertices = np.array(poly_vertices, dtype=np.int32)
    
    # Sử dụng hàm của OpenCV để kiểm tra nếu điểm nằm trong đa giác
    return cv2.pointPolygonTest(poly_vertices, point, False) >= 0  # XEM Note bên trên


def are_all_points_in_polygons(bbox, roi_vertices_list):
    x, y, w, h = bbox
    # Tính tọa độ 4 đỉnh của bounding box
    points = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]

    # Duyệt qua từng ROI trong danh sách roi_vertices_list
    for roi_vertices in roi_vertices_list:
        inside = True
        for point in points:
            # Kiểm tra xem từng điểm có nằm trong ROI hay không
            if not is_point_in_polygon(point, roi_vertices):
                inside = False
                break  # Một điểm không nằm trong ROI, chuyển sang kiểm tra ROI tiếp theo
        if inside:
            return True  # Tất cả các điểm của bounding box nằm trong ROI này
    return False  # Không có ROI nào chứa tất cả các điểm của bounding box


# Hàm cắt ảnh khi tấtt cả các đỉnh box của đối tượng có tất cả các đỉnh nằm trong ROI
def crop_object_in_roi(img, roi_vertices_list, detections, output_dir):
    """
    img dạng numpy 
    detections là list các tọa độ xywh
    """
    # ROI là một danh sách các điểm tạo thành đa giác
    # roi_vertices_list = np.array(roi_vertices, dtype=np.int32)
    
    # # Tạo mask cho vùng ROI
    # mask = np.zeros(img.shape, dtype=np.uint8)
    # cv2.fillConvexPoly(mask, roi_vertices_list, (255, 255, 255))
    
    cropped_images = []
    
    # Duyệt qua tất cả các phát hiện (bounding boxes từ YOLO)
    for detection in detections:
        x, y, w, h = detection  # x, y: tọa độ góc trái, w, h: chiều rộng và chiều cao của bounding box
        bbox=(x, y, w, h)
        # Kiểm tra nếu tất cả các đỉnh của bounding box nằm trong ROI
        if are_all_points_in_polygons(bbox, roi_vertices_list):
            # Cắt ảnh theo bounding box của đối tượng
            cropped_image = img[y:y+h, x:x+w]
            
            # Lấy thời gian hiện tại
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Lưu ảnh cắt vào thư mục output
            #output_path = f"{output_dir}/cropped_{x}_{y}_{int(time.time() * 1000)}.jpg"
            #output_path = f"{output_dir}/cropped_{x}_{y}_{uuid.uuid4().hex}.jpg"
            output_path = f"{output_dir}/cropped_{timestamp}_{uuid.uuid4().hex}.jpg"
            cv2.imwrite(output_path, cropped_image)
            
            cropped_images.append(cropped_image)
    
    return cropped_images

# Hàm cắt ảnh khi tâm - center đối tượng nằm trong ROI
def crop_object_center_in_roi(img, roi_vertices, detections, output_dir):
    # ROI là một danh sách các điểm tạo thành đa giác
    roi_vertices = np.array(roi_vertices, dtype=np.int32)
    
    # Tạo mask cho vùng ROI
    mask = np.zeros(img.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, roi_vertices, (255, 255, 255))
    
    cropped_images = []
    
    # Duyệt qua tất cả các phát hiện (bounding boxes từ YOLO)
    for detection in detections:
        x, y, w, h = detection  # x, y: tọa độ góc trái, w, h: chiều rộng và chiều cao của bounding box
        center = (x + w // 2, y + h // 2)  # Lấy trung tâm của bounding box
        
        # Kiểm tra xem trung tâm của bounding box có nằm trong ROI không
        if is_point_in_polygon(center, roi_vertices):
            # Cắt ảnh theo bounding box của đối tượng
            cropped_image = img[y:y+h, x:x+w]
            
            # Lưu ảnh cắt vào thư mục output
            output_path = f"{output_dir}/cropped_{x}_{y}.jpg"
            cv2.imwrite(output_path, cropped_image)
            
            cropped_images.append(cropped_image)
    
    return cropped_images


