import os
import torch
import numpy as np
import pickle
import cv2
import sys
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1

# --- CẤU HÌNH ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(BASE_DIR, 'module/face_recognition/src'))
from utils import preprocess_for_model
LFW_BIN = os.path.join(BASE_DIR, 'faces_webface_112x112/lfw.bin')
MODEL_DIR = os.path.join(BASE_DIR, 'module/face_recognition/model')
WEIGHTS_PATH = os.path.join(MODEL_DIR, '20180408-102900-casia-webface.pt')

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

def load_bin(path, image_size=(160, 160)):
    with open(path, 'rb') as f:
        # LFW.bin thường chứa [bins, issame_list]
        try:
            bins, issame_list = pickle.load(f, encoding='bytes')
        except:
            bins, issame_list = pickle.load(f)
            
    data_list = []
    for _ in [0, 1]:
        data_list.append(torch.zeros((len(issame_list) * 2, 3, image_size[0], image_size[1])))
        
    for i in range(len(issame_list) * 2):
        _bin = bins[i]
        # Decode ảnh từ byte
        img = cv2.imdecode(np.frombuffer(_bin, np.uint8), -1)
        # Tiền xử lý (Facenet cần is_facenet=True)
        img = cv2.resize(img, image_size)
        img = preprocess_for_model(img, is_facenet=True)
        img = torch.tensor(img, dtype=torch.float32)
        data_list[0][i] = img
        
    print(f"Loaded {len(issame_list)} pairs from {path}")
    return data_list, issame_list

def evaluate(model, data_list, issame_list, batch_size=64):
    model.eval()
    embeddings_list = []
    
    with torch.no_grad():
        for i in range(len(data_list)):
            data = data_list[i]
            embeddings = []
            for j in range(0, data.shape[0], batch_size):
                batch = data[j:j+batch_size].to(DEVICE)
                output = model(batch)
                embeddings.append(output.cpu().numpy())
            embeddings_list.append(np.vstack(embeddings))
            
    # Lấy embeddings của cặp ảnh
    # InsightFace lưu ảnh xen kẽ: [pair1_a, pair1_b, pair2_a, pair2_b, ...]
    # Hoặc lưu 2 list riêng. Ở đây ta giả định list 0 chứa toàn bộ ảnh.
    all_embeddings = embeddings_list[0]
    embeddings1 = all_embeddings[0::2]
    embeddings2 = all_embeddings[1::2]
    
    # Tính Cosine Similarity
    def cos_sim(a, b):
        return np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    
    similarities = cos_sim(embeddings1, embeddings2)
    
    # Tìm ngưỡng tối ưu (Best Threshold)
    best_acc = 0
    best_threshold = 0
    for threshold in np.arange(0, 1, 0.01):
        predict = similarities > threshold
        acc = np.mean(predict == np.array(issame_list))
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
            
    return best_acc, best_threshold

def main():
    print(f"Bắt đầu đánh giá trên thiết bị: {DEVICE}")
    
    # 1. Load Model
    model = InceptionResnetV1(classify=False).to(DEVICE)
    weights = WEIGHTS_PATH
    
    if os.path.exists(weights):
        print(f"Loading weights from {weights}")
        model.load_state_dict(torch.load(weights, map_location=DEVICE), strict=False)
    else:
        print("Warning: No weights found. Evaluating random model.")
        
    # 2. Load Data
    if not os.path.exists(LFW_BIN):
        print(f"Lỗi: Không tìm thấy file {LFW_BIN}")
        return
        
    data_list, issame_list = load_bin(LFW_BIN)
    
    # 3. Chạy đánh giá
    acc, threshold = evaluate(model, data_list, issame_list)
    
    print(f"\n--- KẾT QUẢ ĐÁNH GIÁ LFW ---")
    print(f"Model: {os.path.basename(weights)}")
    print(f"Độ chính xác cao nhất (Best Accuracy): {acc*100:.2f}%")
    print(f"Ngưỡng tối ưu (Best Threshold): {threshold:.2f}")

if __name__ == '__main__':
    main()
