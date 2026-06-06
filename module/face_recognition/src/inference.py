import os
import torch
import numpy as np
import faiss
import pickle
import cv2
try:
    from facenet_pytorch import InceptionResnetV1
except ImportError:
    InceptionResnetV1 = None
from .utils import align_face, preprocess_for_model

class FaceRecognizer:
    """
    Trình trích xuất đặc trưng khuôn mặt (Embedding) bằng MobileFaceNet.
    """
    def __init__(self, weight_path=None, device='cpu', backbone_type='ir_se50'):
        self.device = torch.device(device)
        
        if backbone_type != 'facenet':
            raise ValueError("Only 'facenet' backbone is supported.")
            
        if InceptionResnetV1 is None:
            raise ImportError("Please install facenet-pytorch: pip install facenet-pytorch")
            
        self.model = InceptionResnetV1(classify=False).to(self.device)
        self.embedding_size = 512
            
        self.model.eval()
        
        if weight_path and os.path.exists(weight_path):
            print(f"[FaceRecognizer] Loading {backbone_type} weights from {weight_path}")
            try:
                state_dict = torch.load(weight_path, map_location=self.device)
                # Xử lý các định dạng bọc (extra_static, model, state_dict)
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'model' in state_dict:
                    state_dict = state_dict['model']
                
                msg = self.model.load_state_dict(state_dict, strict=False)
                if len(msg.missing_keys) > 0:
                    print(f"  Missing keys: {len(msg.missing_keys)}")
                if len(msg.unexpected_keys) > 0:
                    print(f"  Unexpected keys: {len(msg.unexpected_keys)}")
            except Exception as e:
                print(f"  Error loading weights: {e}")
        else:
            print(f"Warning: No weights found at {weight_path} for {backbone_type}.")

    def get_embedding(self, img_bgr, landmarks=None):
        """
        Input: Ảnh BGR gốc (OpenCV) và mảng 5 điểm landmarks (tùy chọn)
        Output: Numpy array (1, embedding_size)
        """
        # 1. Căn chỉnh khuôn mặt (Alignment)
        is_facenet = isinstance(self.model, InceptionResnetV1) if InceptionResnetV1 else False
        target_size = (160, 160) if is_facenet else (112, 112)
        
        if landmarks is not None and not is_facenet:
            # Facenet thường dùng MTCNN cắt vuông 160x160 thay vì affine transform 5 điểm của InsightFace
            aligned_img = align_face(img_bgr, landmarks, image_size=target_size)
        else:
            # Fallback hoặc dành cho Facenet (chỉ resize)
            aligned_img = cv2.resize(img_bgr, target_size)
        
        # 2. Tiền xử lý (Convert RGB, Normalize, to Tensor)
        tensor_img = preprocess_for_model(aligned_img, is_facenet=is_facenet)
        
        # Thêm batch dimension -> shape: (1, 3, 112, 112)
        tensor_img = torch.tensor(tensor_img, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 3. Chạy qua model
        with torch.no_grad():
            embedding = self.model(tensor_img)
            embedding = embedding.cpu().numpy()
            
        return embedding


class FaceDatabase:
    """
    Quản lý Vector Database sử dụng thư viện FAISS để tìm kiếm khuôn mặt siêu tốc.
    """
    def __init__(self, index_path='faces.index', users_path='users.pkl', embedding_dim=512):
        self.index_path = index_path
        self.users_path = users_path
        self.embedding_dim = embedding_dim
        
        # Sử dụng IndexFlatIP (Inner Product) cho Cosine Similarity 
        # (Yêu cầu vector phải được chuẩn hóa L2 trước khi đưa vào)
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.users = [] # Mapping ID/Tên
        self.embeddings = [] # Danh sách embeddings để hỗ trợ xóa/rebuild
        
        self.load_database()

    def add_user(self, name, embedding):
        """
        Thêm một người dùng mới vào hệ thống.
        embedding: Numpy array kích thước (1, D)
        """
        # Chuẩn hóa vector trước khi tính Inner Product để có được Cosine Similarity
        faiss.normalize_L2(embedding)
        
        self.index.add(embedding)
        self.users.append(name)
        self.embeddings.append(embedding)
        self.save_database()
        print(f"Added user: {name} successfully.")

    def delete_user(self, index):
        """Xóa người dùng theo vị trí index"""
        if 0 <= index < len(self.users):
            name = self.users[index]
            del self.users[index]
            del self.embeddings[index]
            self.rebuild_index()
            self.save_database()
            print(f"Deleted user: {name} successfully.")
            return True
        return False

    def rebuild_index(self):
        """Xây dựng lại Index từ danh sách embeddings"""
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        if self.embeddings:
            # Gộp các embedding lẻ thành một mảng lớn
            data = np.vstack(self.embeddings)
            self.index.add(data)

    def search(self, embedding, threshold=0.5):
        """
        Tìm kiếm khuôn mặt khớp với embedding đầu vào.
        threshold: Ngưỡng Cosine Similarity (Càng gần 1 càng giống).
        """
        if self.index.ntotal == 0:
            return None, 0.0

        faiss.normalize_L2(embedding)
        
        # k=1: Tìm 1 người gần giống nhất
        distances, indices = self.index.search(embedding, 1)
        
        max_similarity = distances[0][0]
        best_match_idx = indices[0][0]
        
        # Kiểm tra index hợp lệ để tránh IndexError
        if best_match_idx < 0 or best_match_idx >= len(self.users):
            return None, 0.0
            
        if max_similarity > threshold:
            return self.users[best_match_idx], max_similarity
        else:
            return None, max_similarity

    def save_database(self):
        """Lưu lại FAISS Index, danh sách user và embeddings"""
        faiss.write_index(self.index, self.index_path)
        with open(self.users_path, 'wb') as f:
            # Lưu cả users và embeddings vào file pkl
            pickle.dump({'users': self.users, 'embeddings': self.embeddings}, f)

    def load_database(self):
        """Đọc database nếu đã tồn tại"""
        if os.path.exists(self.index_path) and os.path.exists(self.users_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.users_path, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    self.users = data.get('users', [])
                    self.embeddings = data.get('embeddings', [])
                elif isinstance(data, list):
                    # Hỗ trợ định dạng cũ (chỉ có users)
                    self.users = data
                    self.embeddings = []
                    # Nếu list index lớn hơn số lượng user, FAISS sẽ lỗi. Ta cần đảm bảo đồng bộ.
                    if self.index.ntotal > len(self.users):
                        print(f"Warning: Index ({self.index.ntotal}) > Users ({len(self.users)}). Truncating index.")
                        self.rebuild_index_from_zero() # Hoặc chỉ đơn giản là reset
            print(f"Loaded {len(self.users)} users from database.")
        else:
            print("No existing database found. Initialized an empty one.")
