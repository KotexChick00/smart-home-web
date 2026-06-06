import os
import cv2
from uniface import RetinaFace
from inference import FaceRecognizer, FaceDatabase

# --- CẤU HÌNH ---
# Thư mục chứa ảnh các thành viên trong gia đình (Tên file = Tên người)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OWNER_IMAGES_DIR = os.path.join(BASE_DIR, 'module/face_recognition/owner_images')
# Đường dẫn lưu Model & Database
MODEL_DIR = os.path.join(BASE_DIR, 'module/face_recognition/model')
MODEL_WEIGHTS = os.path.join(MODEL_DIR, 'model_ir_se50.pth')
DATABASE_INDEX = os.path.join(MODEL_DIR, 'faces.index')
DATABASE_USERS = os.path.join(MODEL_DIR, 'users.pkl')

def create_db():
    print("Khởi tạo bộ nhận diện...")
    detector = RetinaFace()
    recognizer = FaceRecognizer(weight_path=MODEL_WEIGHTS, backbone_type='ir_se50')
    db = FaceDatabase(index_path=DATABASE_INDEX, users_path=DATABASE_USERS)
    
    if not os.path.exists(OWNER_IMAGES_DIR):
        print(f"Lỗi: Không tìm thấy thư mục {OWNER_IMAGES_DIR}")
        return

    image_files = [f for f in os.listdir(OWNER_IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"Thư mục {OWNER_IMAGES_DIR} đang trống. Hãy bỏ ảnh của bạn vào đây trước.")
        return

    print(f"Tìm thấy {len(image_files)} ảnh chủ nhà. Bắt đầu trích xuất đặc trưng...")

    for filename in image_files:
        name = os.path.splitext(filename)[0]
        img_path = os.path.join(OWNER_IMAGES_DIR, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Lỗi: Không thể đọc ảnh {filename}")
            continue
        
        # 1. Detect khuôn mặt
        faces = detector.detect(img)
        if not faces:
            print(f"Cảnh báo: Không tìm thấy mặt trong ảnh {filename}. Bỏ qua.")
            continue
        
        # Chọn mặt đầu tiên (thường là mặt to nhất/duy nhất)
        face = faces[0]
        
        # 2. Lấy landmarks nếu có (phục vụ Alignment)
        landmarks = None
        if hasattr(face, 'landmarks'):
            landmarks = face.landmarks
        
        # 3. Trích xuất Embedding
        try:
            # Nếu có landmarks -> align_face, nếu không -> fallback resize crop
            # Ở đây ta lấy vùng crop từ bbox để recognizer xử lý
            x1, y1, x2, y2 = map(int, face.bbox)
            face_img = img[max(0, y1):y2, max(0, x1):x2]
            
            embedding = recognizer.get_embedding(face_img, landmarks)
            
            # 4. Thêm vào Database
            db.add_user(name, embedding)
            print(f"Thành công: Đã đăng ký '{name}' vào hệ thống.")
        except Exception as e:
            print(f"Lỗi khi xử lý {filename}: {e}")

    print("\nQuá trình tạo Database hoàn tất!")
    print(f"Số lượng người dùng hiện tại trong hệ thống: {len(db.users)}")

if __name__ == '__main__':
    create_db()
