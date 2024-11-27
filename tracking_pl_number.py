import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
from deep_sort_realtime.deepsort_tracker import DeepSort
from check_ocr_image_matrix import *    # nhận diện chữ số ở biển số xe từ file python check_ocr_image_matrix
from conditions_ocr import *
from roi_crop import are_all_points_in_polygons, crop_object_in_roi
from roi_draw import make_file_name, get_roi_vertices, create_roi_overlay
# Đổi directory về thư mục chứa file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Khởi tạo YOLO và DeepSORT
model = YOLO("models/model_plate.pt")

#model = YOLO("models/yolov8l.pt")
# Chọn thiết bị GPU (cuda) hoặc CPU (cpu)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Sử dụng GPU hoặc CPU để chạy mô hình
model.to(device)  # Di chuyển mô hình lên GPU (nếu có)

tracker = DeepSort(max_age=30, n_init=3, nn_budget=20)
output_dir='roi_image'
# Đường dẫn tới video
video_path = "resources/videos_check_plates/plate1.mp4"

cap = cv2.VideoCapture(video_path)



# Bộ nhớ cache lưu biển số theo track_id
plate_cache = {}
# data_vertices_path=make_file_name(video_path)['path']
roi_vertices= get_roi_vertices(video_path) # lấy giá trị ở hàm gọi từ json
roi_overlay = create_roi_overlay(video_path, color=(0, 255, 0), thickness=2)   # Hàm tạo overlay để vẽ ROI

#print(roi_vertices)
def xyxy_to_xywh(bbox):
    """Chuyển đổi từ format [x1,y1,x2,y2] sang [x,y,w,h]"""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    return [x1, y1, w, h]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    #frame_copy=frame.copy()

    # Nhận diện đối tượng bằng YOLO
    results = model(frame,device=device)
    detections = []
    
    # Xử lý kết quả YOLO và chuyển đổi format bounding box
    for result in results:
        for box in result.boxes:
            # Lấy thông tin bounding box từ YOLO (xyxy format)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())

            if class_id == 0 and confidence<0.2:
                continue
            #Chuyển đổi sang format xywh cho DeepSORT
            xywh = xyxy_to_xywh([x1, y1, x2, y2])
            
            # Lưu thông tin nhận diện vào danh sách
            detections.append((xywh, confidence, class_id))
    
    # Theo dõi đối tượng qua DeepSORT
    tracks = tracker.update_tracks(detections, frame=frame)
    
    # Vẽ bounding box và ID cho các đối tượng được theo dõi
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        
        track_id = track.track_id
        bbox = track.to_ltwh()  # Lấy bounding box dạng np.array([1, 2, 3, 4]) , dạng np.ndarray bbox=np.array([left,top,width,height])
    
        # Chuyển đổi từ ltwh sang tọa độ để vẽ
        x, y = int(bbox[0]), int(bbox[1])
        w, h = int(bbox[2]), int(bbox[3])
        # if 4 đỉnh nằm trong hình thì sẽ cắt ảnh ra
        bbox_type_list_tuple=[(x,y,w,h)]     #lấy x1,y1,x2,y2

        # Nếu chưa nhận diện, cắt vùng ảnh biển số và nhận diện
        plate_image = frame[y:y+h, x:x+w]  # Cắt theo bounding box (y1:y2, x1:x2)
        
        # Kiểm tra xem track_id đã nhận diện chưa
        if track_id not in plate_cache:

            #if 4 đỉnh nằm trong hình thì sẽ cắt ảnh ra
            if are_all_points_in_polygons(bbox,roi_vertices):
            
                # Gọi hàm check() để nhận diện biển số
                #plate_number = recognize_license_plate_mt(plate_image)[0]
                #plate_number = recognize_lp_easyocr(plate_image)[0]
                """
                bbox = track.to_ltwh()  # Lấy bounding box dạng np.array([1, 2, 3, 4]) , dạng np.ndarray bbox=np.array([left,top,width,height])
                    # Chuyển đổi từ ltwh sang tọa độ để vẽ
                x, y = int(bbox[0]), int(bbox[1])
                w, h = int(bbox[2]), int(bbox[3])
                # if 4 đỉnh nằm trong hình thì sẽ cắt ảnh ra
                bbox_type_list_tuple=[(x,y,w,h)]
                """
                crop_object_in_roi(frame, roi_vertices, bbox_type_list_tuple, output_dir)
                #plate_number = "123" # tạm để 123 để test  #recognize_lp_paddleocr(plate_image,padding=10)[0]
                plate_number = recognize_lp_paddleocr(plate_image,padding=10)[0]

                # Lưu biển số vào cache
                plate_cache[track_id] = plate_number  # Lưu biển số vào cache
                print(plate_number)
            else:
                continue
                # Nếu chất lượng không tốt, đặt giá trị mặc định hoặc bỏ qua
                #plate_cache[track_id] = "Unknown"  # hoặc giá trị mặc định khác
        else:
            # Lấy biển số từ cache
            plate_number = plate_cache[track_id]


        # Vẽ bounding box và thông tin lên frame (nếu bạn muốn hiển thị trực tiếp lên màn hình)
        color = (0, 255, 0)  # Màu sắc của bounding box (màu xanh lá)
        thickness = 2  # Độ dày của đường viền bounding box
        label = f"ID: {track_id} Conf: {confidence:.2f} plate: {plate_number}"

        # Vẽ bounding box lên frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        # Vẽ label lên bounding box
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    frame_with_roi = cv2.addWeighted(frame, 1.0, roi_overlay, 1.0, 0)   # vẽ ROI lên frame, đã xử lý ở bên ngoài vòng while rồi nên nhẹ

    # Hiển thị khung hình
    frame=frame_with_roi
    # Hiển thị video
    cv2.imshow("Tracking", frame)
    
    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()