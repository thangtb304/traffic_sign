
from easyROI.EasyROI.easyROI import EasyROI
import cv2
from pprint import pprint
import json
from urllib.parse import urlparse, parse_qs
import numpy as np
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def make_file_name(rtsp_name):   # hàm này bên dev giao diện KHÔNG CẦN DÙNG
    """
    input:
   
    output:
    Hàm này sẽ tạo ra tên file (base_name) chứa data roi và đường dẫn (roi_data_path) tới thư mục chứa file roi
    """
    # Phân tích URL
    parsed_url = urlparse(rtsp_name)
    # Lấy tên máy chủ và kênh từ tham số query
    host = parsed_url.hostname
    port = parsed_url.port
    query_params = parse_qs(parsed_url.query)
    channel = query_params.get("channel", ["unknown"])[0]
    subtype = query_params.get("subtype", ["unknown"])[0]

    # Tạo tên cơ bản từ link rtsp
    base_name = f"{host}_{port}_channel_{channel}_subtype_{subtype}"

    filename = f"{base_name}.json"
    # Đường dẫn đầy đủ tới file trong thư mục roi_data
    output_dir = "roi_data"
    roi_data_path = os.path.join(output_dir, filename)

    # Kiểm tra và tạo thư mục nếu chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_name={}
    file_name['name']=base_name         # tên cơ bản của file (không tính tên đuôi file)
    file_name['path']=roi_data_path   # đường dẫn chứa file json lưu thông tin tọa độ
    return file_name

def convert_intc_2_int(polygon_roi):  # hàm này bên dev giao diện KHÔNG CẦN DÙNG
    """
    Hàm này để chuyển tọa độ các đỉnh của polygon lúc vẽ từ định dạng intc sang int để lưu được trong file json
    Nếu không chuyển thì không lưu được trong file json
    Khi chưa chuyển thì các số ở dạng np.int32(0), ví dụ số 0 thì phải biểu diễn là np.int32(0)
    dạng tọa độ thì vẫn lưu dưới dạng list các điểm tọa độ, không phải dạng array
    Polygon Example:
        h= {'roi': {0: {'vertices': [(744, 88),
                                (1013, 176),
                                (781, 423),
                                (468, 386),
                                (510, 229)]}},    # đây là dạng list, không phải array
            'type': 'polygon'}
            k = h['roi'][np.int32(0)]['vertices']    đây là dạng list chứa các số dạng np.int32
    """
    converted_roi = {
        'type': polygon_roi['type'],
        'roi': {}
    }
    
    for key, value in polygon_roi['roi'].items():
        converted_vertices = [(int(x), int(y)) for x, y in value['vertices']]
        converted_roi['roi'][int(key)] = {'vertices': converted_vertices}
    return converted_roi

def save_data(dict_roi,video_path):   # hàm này bên dev giao diện KHÔNG CẦN DÙNG
    """
    input: 
    + dict_roi chứa thông tin các đỉnh của hình vẽ, 
    + video_path: đường dẫn camera: rtsp_name

    output:
    lưu thông tin tọa độ vào file json nằm trong tệp "roi_data"
    """
    data_vertices_path = make_file_name(video_path)['path']
    
    # Kiểm tra nếu file không tồn tại hoặc trống, tạo một list rỗng
    if not os.path.exists(data_vertices_path) or os.stat(data_vertices_path).st_size == 0:
        data = []
    else:
        # Nếu file tồn tại và có dữ liệu, đọc dữ liệu
        with open(data_vertices_path, 'r+', encoding='utf-8') as f:
            data = json.load(f)
    
    # Thêm dữ liệu mới vào danh sách
    data.append(dict_roi)
    
    # Ghi dữ liệu trở lại vào file
    with open(data_vertices_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def add_function(name_function,video_path):  # hàm này bên dev giao diện dùng để nối API
    """
    input:
    + name_function: tên hàm/tên loại hình vẽ mà người dùng muốn vẽ.
    ---draw_line: người dùng sẽ vẽ đường thẳng
    ---draw_rectangle: người dùng sẽ vẽ hình chữ nhật
    ---draw_polygon: người dùng vẽ tam giác, đa giác. mỗi lần vẽ thì thực hiện ấn 1 click chuột trái. Khi nào muốn hoàn thành hình vẽ
    thì ấn đúp chuột trái
    ---draw_circle: người dùng vẽ đường tròn.

    """
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), 'Cannot capture source'
    ret, frame = cap.read()
    roi_helper = EasyROI(verbose=True)

    #2. Load hình cũ đã vẽ
    data_vertices = make_file_name(video_path)['path']
        # Kiểm tra nếu file không tồn tại hoặc trống, tạo một list rỗng
    if not os.path.exists(data_vertices) or os.stat(data_vertices).st_size == 0:
        data = []
    else:
        # Nếu file tồn tại và có dữ liệu, đọc dữ liệu
        with open(data_vertices, 'r+', encoding='utf-8') as f:
            data = json.load(f)

    # Khởi tạo một ảnh gốc       
    frame_temp = frame.copy()  # Dùng ảnh gốc hoặc tạo ảnh mới từ frame gốc
    for i in data:
        # if i["camera"]!=video_path:
        #     continue
        dict_roi=i

        # Khởi tạo một ảnh mới
        frame_temp = roi_helper.visualize_roi(frame_temp, dict_roi)  # Vẽ ROI vào ảnh này
       
    #cv2.imshow("frame", frame_temp)

    frame=frame_temp

    if hasattr(roi_helper, name_function):
        # Gọi hàm tương ứng với tên hàm truyền vào
        func = getattr(roi_helper, name_function)
    # DRAW RECTANGULAR ROI
    dict_roi = func(frame, 1)

    if name_function == "draw_polygon" and "type" in dict_roi and dict_roi["type"] is not None:
        dict_roi = convert_intc_2_int(dict_roi)
        # print("Rectangle Example:")
        # pprint(dict_roi)

    frame_temp = roi_helper.visualize_roi(frame, dict_roi)
    cv2.imshow("frame", frame_temp)
    key = cv2.waitKey(0)
    if key & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    dict_roi['camera']=video_path
    if "type" in dict_roi and dict_roi["type"] is not None:
        save_data(dict_roi,video_path)
    #save_data(dict_roi,video_path)
    return dict_roi


def only_view(video_path):  # # hàm này bên dev giao diện dùng để nối API
    """Hàm này sẽ hiển thị tất cả các hình vẽ mà người dùng đã vẽ"""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), 'Cannot capture source'
    ret, frame = cap.read()
    roi_helper = EasyROI(verbose=True)

    ## Load data from file json
    data_vertices = make_file_name(video_path)['path']
        # Kiểm tra nếu file không tồn tại hoặc trống, tạo một list rỗng
    if not os.path.exists(data_vertices) or os.stat(data_vertices).st_size == 0:
        data = []
    else:
        # Nếu file tồn tại và có dữ liệu, đọc dữ liệu
        with open(data_vertices, 'r+', encoding='utf-8') as f:
            data = json.load(f)

    # Khởi tạo một ảnh gốc       
    frame_temp = frame.copy()  # Dùng ảnh gốc hoặc tạo ảnh mới từ frame gốc
    for i in data:
        if i["camera"]!=video_path:
            continue
        dict_roi=i
        # Khởi tạo một ảnh mới
        frame_temp = roi_helper.visualize_roi(frame_temp, dict_roi)  # Vẽ ROI vào ảnh này
        #frame_temp = roi_helper.visualize_roi(frame, dict_roi)
    cv2.imshow("frame", frame_temp)
    key = cv2.waitKey(0)
    if key & 0xFF == ord('q'):
        cv2.destroyAllWindows()

def delete_last(video_path):   # hàm này bên dev giao diện dùng để nối API
    """
    Hàm này sẽ xóa đi hình vẽ mới nhất
    """
    data_vertices = make_file_name(video_path)['path']
    
    # Kiểm tra nếu file không tồn tại hoặc trống, tạo một list rỗng
    if not os.path.exists(data_vertices) or os.stat(data_vertices).st_size == 0:
        data = []
    else:
        # Nếu file tồn tại và có dữ liệu, đọc dữ liệu
        with open(data_vertices, 'r+', encoding='utf-8') as f:
            data = json.load(f)   
    # Thêm dữ liệu mới vào danh sách
    data.pop()    
    # Ghi dữ liệu trở lại vào file
    with open(data_vertices, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    only_view(video_path)

def get_roi_vertices(video_path):
    """
    Hàm này lấy thông tin các điểm của các đỉnh của polygon. ví dụ tập hợp các đỉnh là:
    list của [[549,463],[916,467],[920,613],[515,613]]
    """
    data_vertices_path=make_file_name(video_path)['path']
    if not os.path.exists(data_vertices_path) or os.stat(data_vertices_path).st_size == 0:
        data = []
    else:
        with open(data_vertices_path,"r",encoding="utf-8") as f:
            data=json.load(f)
    list_vertices=[]
    for i in data:
        if i['type']=="polygon":
            k=i['roi']['0']['vertices']
            list_vertices.append(k)
    return list_vertices

# Hàm tạo overlay để vẽ ROI
# def create_roi_overlay(video_path, roi_vertices, color=(0, 255, 0), thickness=2):
#     """
#     Tạo một overlay chứa ROI.
    
#     :param roi_vertices: Danh sách các điểm của ROI.
#     :param color: Màu vẽ (BGR).
#     :param thickness: Độ dày nét vẽ.
#     :return: roi_overlay (numpy array).
#     """
#     cap = cv2.VideoCapture(video_path)

#     # Lấy thông tin video
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # Tạo lớp overlay với ROI
#     frame_size = (frame_height, frame_width, 3) #:param frame_size: Kích thước khung hình (width, height, channels).

#     roi_overlay = np.zeros(frame_size, dtype=np.uint8)  # Tạo lớp overlay trống
#     roi_vertices = np.array([roi_vertices], dtype=np.int32)  # Chuyển ROI vertices thành numpy array
#     cv2.polylines(roi_overlay, roi_vertices, isClosed=True, color=color, thickness=thickness)  # Vẽ ROI lên overlay
#     cap.release()
#     cv2.destroyAllWindows()
#     return roi_overlay


########
def create_roi_overlay(video_path, color=(0, 255, 0), thickness=2):
    """
    Tạo một overlay chứa ROI.
    
    :param roi_vertices_list: Danh sách các list, mỗi 1 đối tượng của list là 1 hình đa giác ROI.
    :param color: Màu vẽ (BGR).
    :param thickness: Độ dày nét vẽ.
    :return: roi_overlay (numpy array).
    """
    roi_vertices_list=get_roi_vertices(video_path)
    cap = cv2.VideoCapture(video_path)

    # Lấy thông tin video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Tạo lớp overlay với ROI
    frame_size = (frame_height, frame_width, 3) #:param frame_size: Kích thước khung hình (width, height, channels).

    roi_overlay = np.zeros(frame_size, dtype=np.uint8)  # Tạo lớp overlay trống

    # Duyệt qua tất cả các ROI và vẽ từng đa giác lên overlay
    for roi_vertices in roi_vertices_list:
        roi_vertices = np.array([roi_vertices], dtype=np.int32)  # Chuyển ROI vertices thành numpy array
        cv2.polylines(roi_overlay, roi_vertices, isClosed=True, color=color, thickness=thickness)  # Vẽ ROI lên overlay
    
    cap.release()
    cv2.destroyAllWindows()
    return roi_overlay

# def create_roi_overlay(frame_size, roi_vertices, color=(0, 255, 0), thickness=2):
#     """
#     Tạo một overlay chứa ROI.
#     :param frame_size: Kích thước khung hình (width, height, channels).
#     :param roi_vertices: Danh sách các điểm của ROI.
#     :param color: Màu vẽ (BGR).
#     :param thickness: Độ dày nét vẽ.
#     :return: overlay (numpy array).
#     """
#     overlay = np.zeros(frame_size, dtype=np.uint8)  # Tạo lớp overlay trống
#     roi_vertices = np.array([roi_vertices], dtype=np.int32)  # Chuyển ROI vertices thành numpy array
#     cv2.polylines(overlay, roi_vertices, isClosed=True, color=color, thickness=thickness)  # Vẽ ROI lên overlay
#     return overlay
if __name__=="__main__":


    # video_path = 'input/overpass.mp4'
    video_path = "resources/videos_check_plates/plate1.mp4"
    #video_path = "resources/videos_check_plates/kiemtra.mp4"

    #Test hàm ở bên dưới nhé:
    # print("-----")
    # h=get_roi_vertices(video_path)
    # #k=h[0]['roi']['0']['vertices']
    # # # print(k)
    # # # print(type(k))
    # # #print(k)
    # print(h[0])
    # print(type(h[0]))

    # 1. Hàm vẽ chữ nhật, đường thẳng, tròn, đa giác
    #add_function("draw_rectangle",video_path)
    #add_function("draw_line",video_path)
    #add_function("draw_circle",video_path)
    add_function("draw_polygon",video_path)

    # 2. Hàm hiển thị các hình đã vẽ
    #only_view(video_path)

    # 3. Hàm xóa hình vẽ cuối cùng (nếu muốn xóa tất cả các hình đã vẽ thì phải làm nhiều lần hàm delete_last)
    # delete_last(video_path)


    # def hiih():
    #     print('hiiiii')
