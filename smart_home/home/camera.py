import cv2
import time
import os
import threading
import numpy as np
import pickle

# --- KHỞI TẠO HỆ THỐNG LƯU TRỮ CỤC BỘ ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'face_data')
MODEL_PATH = os.path.join(BASE_DIR, 'lbph_model.yml')
LABEL_PATH = os.path.join(BASE_DIR, 'labels.pkl')

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Dùng Haar Cascade có sẵn của OpenCV (nhẹ, nhanh, không cần file ngoài)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Nạp dữ liệu tên nếu đã từng đăng ký trước đó
label_dict = {}
if os.path.exists(MODEL_PATH) and os.path.exists(LABEL_PATH):
    recognizer.read(MODEL_PATH)
    with open(LABEL_PATH, 'rb') as f:
        label_dict = pickle.load(f)

class CameraStreamer:
    def __init__(self):
        self.cap = None
        self.frame = None
        self.processed_frame = None
        self.detect_only_frame = None
        self.running = False
        self.lock = threading.Lock()
        
        self.last_face_detected_time = 0
        self.recognized_user = None
        
        self.start()

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()

    def stop(self):
        if self.running:
            self.running = False
            # Chờ luồng thread kết thúc để giải phóng Cap một cách an toàn
            if hasattr(self, 'thread'):
                self.thread.join(timeout=1)
            with self.lock:
                self.frame = None
                self.processed_frame = None
                self.detect_only_frame = None

    def _update(self):
        print("[Camera] Hệ thống Nhận diện Khuôn mặt (LBPH) đã khởi động.")
        self.cap = cv2.VideoCapture(0)
        
        while self.running:
            success, frame = self.cap.read()
            if not success:
                time.sleep(0.1)
                continue

            with self.lock:
                self.frame = frame.copy()
            
            # Chuyển sang ảnh xám để tăng tốc độ xử lý
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
            
            display_frame = frame.copy()
            detect_only = frame.copy()
            
            best_user = None
            
            for (x, y, w, h) in faces:
                # 1. Giao diện trang Đăng ký (Khung xanh dương)
                cv2.rectangle(detect_only, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                roi_gray = gray[y:y+h, x:x+w]
                label = "Unknown"
                color = (0, 0, 255) # Màu đỏ cho Unknown
                
                # 2. Giao diện trang Đăng nhập (Nhận diện)
                if len(label_dict) > 0:
                    id_, conf = recognizer.predict(roi_gray)
                    # LBPH: Khoảng cách (Confidence) càng nhỏ càng giống. Ngưỡng tốt thường < 75.
                    user_name = label_dict.get(id_)
                    if conf < 75 and user_name:
                        label = f"{user_name} ({int(conf)})"
                        color = (0, 255, 0) # Xanh lá nếu nhận diện đúng người trong db
                        best_user = user_name
                    else:
                        label = f"Unknown ({int(conf)})" if conf < 75 else "Unknown"
                        color = (0, 0, 255) # Màu đỏ nếu không khớp hoặc không tìm thấy trong db
                
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(display_frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Cập nhật trạng thái để trang Web biết có mở nút Đăng nhập hay không
            if best_user:
                self.last_face_detected_time = time.time()
                self.recognized_user = best_user

            ret1, buffer1 = cv2.imencode('.jpg', display_frame)
            ret2, buffer2 = cv2.imencode('.jpg', detect_only)
            
            with self.lock:
                if ret1: self.processed_frame = buffer1.tobytes()
                if ret2: self.detect_only_frame = buffer2.tobytes()
            
            time.sleep(0.04)

        # Giải phóng phần cứng webcam khi vòng lặp dừng
        if self.cap:
            self.cap.release()
            self.cap = None
        print("[Camera] Đã tạm dừng Camera và giải phóng phần cứng thành công.")

streamer = CameraStreamer()

def gen_frames(mode='recognition'):
    while True:
        with streamer.lock:
            frame_bytes = streamer.detect_only_frame if mode == 'register' else streamer.processed_frame
        
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04)

def get_recognition_status():
    return streamer.last_face_detected_time, streamer.recognized_user

def register_new_user(name):
    """Hàm chụp ảnh và huấn luyện AI tại chỗ"""
    with streamer.lock:
        frame = streamer.frame.copy() if streamer.frame is not None else None
    
    if frame is None:
        return False, "Chưa bật Camera."
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    
    if len(faces) == 0:
        return False, "Không tìm thấy khuôn mặt! Hãy nhìn thẳng vào Camera."
        
    # Lấy khuôn mặt to và gần Camera nhất để làm mẫu đăng ký
    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
    x, y, w, h = faces[0]
    roi_gray = gray[y:y+h, x:x+w]
    
    # Tạo ID duy nhất cho người dùng mới
    new_id = 0 if len(label_dict) == 0 else max(label_dict.keys()) + 1
    label_dict[new_id] = name
    
    # Tạo biến thể dữ liệu (Data Augmentation) để AI học chuẩn hơn từ 1 bức ảnh
    faces_data = [roi_gray, cv2.flip(roi_gray, 1)] 
    ids = np.array([new_id, new_id])
    
    # Huấn luyện mô hình ngay lập tức
    if os.path.exists(MODEL_PATH):
        recognizer.update(faces_data, ids)
    else:
        recognizer.train(faces_data, ids)
        
    # Lưu trí nhớ xuống ổ cứng
    recognizer.write(MODEL_PATH)
    with open(LABEL_PATH, 'wb') as f:
        pickle.dump(label_dict, f)
        
    return True, f"Đã đăng ký thành công ID Face: {name}"

def delete_user(user_id):
    """Xóa người dùng khỏi danh sách được phép truy cập (label_dict)"""
    global label_dict
    if user_id in label_dict:
        label_dict.pop(user_id)
        try:
            with open(LABEL_PATH, 'wb') as f:
                pickle.dump(label_dict, f)
            return True
        except Exception as e:
            print("Lỗi khi ghi lại labels.pkl:", e)
            return False
    return False