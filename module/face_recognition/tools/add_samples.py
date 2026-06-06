import os
import cv2
import random
import sys
from uniface import RetinaFace
from inference import FaceRecognizer, FaceDatabase

# --- CẤU HÌNH ---
# Thư mục chứa ảnh thô đã trích xuất
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(BASE_DIR, 'faces_webface_112x112_raw')
MODEL_DIR = os.path.join(BASE_DIR, 'module/face_recognition/model')
MODEL_WEIGHTS = os.path.join(MODEL_DIR, 'model_ir_se50.pth')
DATABASE_INDEX = os.path.join(MODEL_DIR, 'faces.index')
DATABASE_USERS = os.path.join(MODEL_DIR, 'users.pkl')

NUM_SAMPLES = 10 # Số lượng người dùng ngẫu nhiên muốn nạp

def add_samples():
    print("Khởi tạo hệ thống...")
    detector = RetinaFace()
    recognizer = FaceRecognizer(weight_path=MODEL_WEIGHTS, backbone_type='ir_se50')
    db = FaceDatabase(index_path=DATABASE_INDEX, users_path=DATABASE_USERS)
    
    if not os.path.exists(DATA_DIR):
        print(f"Lỗi: Không tìm thấy thư mục dataset tại {DATA_DIR}")
        return

    # Lấy danh sách tất cả các thư mục identity
    identities = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    if len(identities) < NUM_SAMPLES:
        NUM_TO_PICK = len(identities)
    else:
        NUM_TO_PICK = NUM_SAMPLES
        
    selected_identities = random.sample(identities, NUM_TO_PICK)
    print(f"Bắt đầu nạp {NUM_TO_PICK} identities ngẫu nhiên vào Database...")

    for identity in selected_identities:
        identity_path = os.path.join(DATA_DIR, identity)
        images = [f for f in os.listdir(identity_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not images: continue
        
        # Chọn 1 ảnh bất kỳ của identity đó
        img_filename = random.choice(images)
        img_path = os.path.join(identity_path, img_filename)
        img = cv2.imread(img_path)
        
        if img is None: continue
        
        # Vì ảnh trong CASIA-WebFace thường đã được alignment sẵn (nếu là bản 112x112)
        # Nhưng ta vẫn chạy detect để lấy bbox/landmarks nếu cần
        faces = detector.detect(img)
        
        landmarks = None
        if faces:
            # Nếu detect ra mặt thì lấy landmarks
            face = faces[0]
            landmarks = getattr(face, 'landmarks', None)
        
        # Trích xuất Embedding
        embedding = recognizer.get_embedding(img, landmarks)
        
        # Thêm vào DB với tên Sample_ID
        name = f"Sample_{identity}"
        db.add_user(name, embedding)
        print(f"Đã nạp: {name}")

    print("\nHoàn tất nạp mẫu!")
    print(f"Tổng số người dùng trong database: {len(db.users)}")

if __name__ == '__main__':
    add_samples()
