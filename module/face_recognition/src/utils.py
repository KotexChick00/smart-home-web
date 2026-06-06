import cv2
import numpy as np

# Các điểm mốc chuẩn (Reference Facial Landmarks) cho ảnh 112x112
# Nguồn: InsightFace (ArcFace)
REFERENCE_FACIAL_POINTS = np.array([
    [38.2946, 51.6963], # Mắt trái
    [73.5318, 51.5014], # Mắt phải
    [56.0252, 71.7366], # Mũi
    [41.5493, 92.3655], # Mép trái
    [70.7299, 92.2041]  # Mép phải
], dtype=np.float32)

def align_face(img, landmarks, image_size=(112, 112)):
    """
    Hàm căn chỉnh khuôn mặt dựa trên 5 điểm landmarks.
    
    Args:
        img: Ảnh gốc (numpy array BGR).
        landmarks: Danh sách 5 điểm landmarks [[x1, y1], [x2, y2], ...].
                   Thứ tự: Mắt trái, Mắt phải, Mũi, Mép trái, Mép phải.
        image_size: Kích thước ảnh đầu ra, mặc định (112, 112).
        
    Returns:
        Ảnh khuôn mặt đã được crop và căn chỉnh.
    """
    if isinstance(landmarks, dict):
        # Chuyển đổi dict sang list nếu thư viện trả về dạng dict
        src_pts = np.array([
            landmarks.get('left_eye', [0,0]),
            landmarks.get('right_eye', [0,0]),
            landmarks.get('nose', [0,0]),
            landmarks.get('mouth_left', [0,0]),
            landmarks.get('mouth_right', [0,0])
        ], dtype=np.float32)
    else:
        # Giả định đã truyền đúng mảng 5x2
        src_pts = np.array(landmarks, dtype=np.float32)

    # Nếu không đủ 5 điểm, sử dụng crop bounding box thông thường để fallback
    if src_pts.shape[0] < 5:
        return cv2.resize(img, image_size)

    # Tính toán ma trận biến đổi Affine (xoay, scale, tịnh tiến)
    # estimateAffinePartial2D tìm ra phép biến đổi tối ưu giữ nguyên tỷ lệ
    tform, inliers = cv2.estimateAffinePartial2D(src_pts, REFERENCE_FACIAL_POINTS)
    
    # Nếu không tìm được ma trận, trả về ảnh resize bình thường
    if tform is None:
        return cv2.resize(img, image_size)

    # Áp dụng ma trận biến đổi lên ảnh gốc
    aligned_face = cv2.warpAffine(img, tform, image_size)
    
    return aligned_face

def preprocess_for_model(aligned_img, is_facenet=False):
    """
    Tiền xử lý ảnh trước khi đưa vào PyTorch Model.
    """
    # Convert BGR (OpenCV) sang RGB
    rgb_img = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB)
    
    if is_facenet:
        # facenet-pytorch standardization: (img - 127.5) / 128.0
        img_tensor = (rgb_img - 127.5) / 128.0
    else:
        # Scale pixel value từ [0, 255] về [-1, 1] (Chuẩn của InsightFace/MobileFaceNet)
        img_tensor = (rgb_img / 255.0 - 0.5) / 0.5
    
    # Đổi shape từ (H, W, C) sang (C, H, W)
    img_tensor = np.transpose(img_tensor, (2, 0, 1))
    
    return img_tensor
