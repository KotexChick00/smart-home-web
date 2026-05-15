import cv2
import time
import os
import sys
import threading
from django.conf import settings
from uniface import RetinaFace

# Sửa lỗi OMP: Error #15 trên macOS khi dùng nhiều thư viện AI
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Thêm đường dẫn module face_recognition vào sys.path để import inference
FR_SRC_PATH = os.path.join(settings.BASE_DIR.parent, 'module', 'face_recognition', 'src')
if FR_SRC_PATH not in sys.path:
    sys.path.append(FR_SRC_PATH)

from inference import FaceRecognizer, FaceDatabase

# --- KHỞI TẠO SINGLETON AI ---
detector = RetinaFace()
MODEL_DIR = os.path.join(settings.BASE_DIR.parent, 'module', 'face_recognition', 'model')

# Cập nhật đường dẫn cho ResNet-18 (MS1MV3)
FINETUNED_PATH = os.path.join(MODEL_DIR, 'finetuned', 'backbone_ir_se18_epoch_3.pth')
DEFAULT_PATH = os.path.join(MODEL_DIR, 'MS1MV3_arcface_r18_p16.pth')
WEIGHTS_PATH = FINETUNED_PATH if os.path.exists(FINETUNED_PATH) else DEFAULT_PATH

print(f"[Camera Debug] MODEL_DIR: {MODEL_DIR}")
print(f"[Camera Debug] FINETUNED_PATH: {FINETUNED_PATH} (Exists: {os.path.exists(FINETUNED_PATH)})")
print(f"[Camera Debug] DEFAULT_PATH: {DEFAULT_PATH} (Exists: {os.path.exists(DEFAULT_PATH)})")
print(f"[Camera Debug] CHOSEN WEIGHTS: {WEIGHTS_PATH}")

recognizer = FaceRecognizer(weight_path=WEIGHTS_PATH, backbone_type='ir_se18')
db = FaceDatabase(
    index_path=os.path.join(MODEL_DIR, 'faces.index'),
    users_path=os.path.join(MODEL_DIR, 'users.pkl')
)

class CameraStreamer:
    """Hệ thống quản lý Camera tập trung để tránh xung đột tài nguyên"""
    def __init__(self):
        self.camera_index = 0
        self.cap = None
        self.frame = None
        self.processed_frame = None
        self.running = False
        self.lock = threading.Lock()
        
        # Gắn kết AI module vào streamer
        self.db = db
        self.recognizer = recognizer
        
        # Biến cho Recognition
        self.last_face_detected_time = 0
        self.recognized_user = None
        
        # Khởi động thread
        self.start()

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()

    def set_camera_index(self, index):
        with self.lock:
            if self.camera_index != index:
                print(f"[Camera] Chuyển sang index: {index}")
                self.camera_index = index
                if self.cap:
                    self.cap.release()
                    self.cap = None

    def _update(self):
        print("[Camera] Background thread started.")
        frame_count = 0
        process_every_n_frames = 3
        faces = []
        
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.camera_index)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                time.sleep(1) # Chờ camera khởi động
                continue

            success, frame = self.cap.read()
            if not success:
                time.sleep(0.1)
                continue

            # Lưu frame gốc để đăng ký
            with self.lock:
                self.frame = frame.copy()
            
            # AI Processing (Detect)
            frame_count += 1
            if frame_count % process_every_n_frames == 0:
                faces = detector.detect(frame)
            
            # Vẽ UI nhận diện lên một bản copy khác để stream
            display_frame = frame.copy()
            if faces:
                current_time = time.time()
                h, w = frame.shape[:2]
                
                best_similarity = -1
                best_user = None
                
                for face in faces:
                    x1, y1, x2, y2 = map(int, face.bbox)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        face_crop = frame[y1:y2, x1:x2]
                        landmarks = getattr(face, 'landmarks', None)
                        
                        # AI Recognition
                        embedding = self.recognizer.get_embedding(face_crop, landmarks)
                        user_name, similarity = self.db.search(embedding, threshold=0.8)
                        
                        if user_name:
                            color = (0, 255, 0)
                            label = f"{user_name} ({similarity:.2f})"
                            
                            # Lưu người có độ tương đồng cao nhất để phục vụ Login
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_user = user_name
                        else:
                            color = (0, 0, 255)
                            label = "Unknown"

                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Cập nhật trạng thái người dùng nhận diện được (để hiện nút Login)
                if best_user:
                    self.last_face_detected_time = current_time
                    self.recognized_user = best_user

            # Lưu frame đã vẽ UI để stream
            ret, buffer = cv2.imencode('.jpg', display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ret:
                with self.lock:
                    self.processed_frame = buffer.tobytes()
            
            # Khống chế FPS của background thread
            time.sleep(0.01)

# Singleton Instance
streamer = CameraStreamer()

def gen_frames():
    while True:
        with streamer.lock:
            frame_bytes = streamer.processed_frame
        
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04) # ~25 FPS cho stream

def get_recognition_status():
    return streamer.last_face_detected_time, streamer.recognized_user

def register_new_user(name):
    with streamer.lock:
        frame = streamer.frame.copy() if streamer.frame is not None else None
    
    if frame is None:
        return False, "Không có dữ liệu hình ảnh. Vui lòng chờ camera khởi động."
    
    faces = detector.detect(frame)
    if not faces:
        return False, "Không tìm thấy khuôn mặt trong khung hình."
    
    face = faces[0]
    landmarks = getattr(face, 'landmarks', None)
    x1, y1, x2, y2 = map(int, face.bbox)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    face_img = frame[y1:y2, x1:x2]
    
    try:
        embedding = streamer.recognizer.get_embedding(face_img, landmarks)
        streamer.db.add_user(name, embedding)
        return True, f"Đã đăng ký thành công: {name}"
    except Exception as e:
        return False, f"Lỗi: {str(e)}"

def set_camera_index(index):
    streamer.set_camera_index(int(index))
