import cv2
import time
import os
from django.conf import settings
from uniface import RetinaFace

# Tự động tải khởi tạo model RetinaFace (chạy lần đầu sẽ download model)
detector = RetinaFace()

# Tạo sẵn thư mục lưu ảnh
FACE_DIR = os.path.join(settings.BASE_DIR, 'face_detection')
os.makedirs(FACE_DIR, exist_ok=True)

# Parameter để chống spam hình ảnh
last_save_time = 0
SAVE_INTERVAL = 2.0  # Chỉ lưu nếu cách frame liền trước >= 2s

def gen_frames():
    """Generator function yield các frame hình ảnh thành luồng (MJPEG)."""
    global last_save_time
    # Khởi động webcam mặc định (thường số 0 là camera laptop)
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # 1. Chạy AI nhận diện khuôn mặt
            faces = detector.detect(frame)
            
            # 2. Xử lý lưu khuôn mặt (Nhiệm vụ UniFace)
            if faces:
                current_time = time.time()
                # Kiểm tra delay để không spam đầy ổ cứng
                if current_time - last_save_time > SAVE_INTERVAL:
                    # Rút trích khuôn mặt đầu tiên (to nhất)
                    face = faces[0]
                    # Thu được Box (X1, Y1, X2, Y2)
                    x1, y1, x2, y2 = map(int, face.bbox)
                    
                    # Cắt khung hình (Crop bounding box) - phải check biên
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        face_crop = frame[y1:y2, x1:x2]
                        filename = f"face_{int(current_time)}.jpg"
                        filepath = os.path.join(FACE_DIR, filename)
                        cv2.imwrite(filepath, face_crop)
                        print(f"[AI] Đã phát hiện và lưu khuôn mặt: {filename}")
                        last_save_time = current_time
            
            # 3. Vẽ khung đỏ hiển thị lên Web
            for face in faces:
                x1, y1, x2, y2 = map(int, face.bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Encode về chuỗi byte để gửi HTTP
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
