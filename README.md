# Smart Home AIoT Gateway

Hệ thống Nhà thông minh tích hợp Trí tuệ Nhân tạo (AIoT) và Thị giác Máy tính (Computer Vision), được thiết kế với kiến trúc Edge Computing. Dự án kết hợp phần cứng Yolo:Bit, giao thức truyền thông MQTT và nền tảng Web Django để tự động hóa việc quản lý năng lượng dựa trên dữ liệu môi trường thực tế.

## 🌟 Tính năng cốt lõi

1. **Giao tiếp IoT thời gian thực:** Sử dụng giao thức MQTT (qua HiveMQ Broker) để đồng bộ dữ liệu cảm biến (Nhiệt độ, Độ ẩm, Ánh sáng) và lệnh điều khiển thiết bị (Đèn, Quạt) giữa phần cứng và Web Dashboard.
2. **AI Automation - Tối ưu hóa Năng lượng:**
   * **Occupancy Detection:** Tự động bật/tắt đèn dựa trên mô hình phân loại (RandomForest) học từ tập dữ liệu phát hiện sự hiện diện của con người.
   * **Thermal Comfort:** Tự động điều khiển quạt dựa trên mô hình hồi quy tính toán chỉ số cảm giác nhiệt (PMV) từ bộ dữ liệu tiêu chuẩn quốc tế ASHRAE.
3. **Smart Face ID (LBPH):** Hệ thống xác thực người dùng không cần mật khẩu bằng nhận diện khuôn mặt cục bộ, ứng dụng thuật toán LBPH và Haar Cascades của OpenCV.
4. **Web Dashboard Động:** Giao diện điều khiển trung tâm được xây dựng bằng Django, cập nhật trạng thái thiết bị thời gian thực thông qua API và AJAX.

## 📁 Cấu trúc thư mục

```text
📦 smart-home-web
 ┣ 📂 .venv/                  # Môi trường ảo Python
 ┗ 📂 smart_home/             # Thư mục gốc của dự án Django
   ┣ 📂 home/                 # Ứng dụng Web chính (Django App)
   ┃ ┣ 📂 face_data/          # Chứa dữ liệu khuôn mặt đã đăng ký (Tự động tạo)
   ┃ ┣ 📂 migrations/         # Các file đồng bộ cơ sở dữ liệu
   ┃ ┣ 📂 templates/          # Chứa các file giao diện HTML (login, dashboard...)
   ┃ ┣ 📜 admin.py
   ┃ ┣ 📜 apps.py             # Cấu hình khởi động luồng MQTT ngầm
   ┃ ┣ 📜 camera.py           # Module xử lý Computer Vision (Face ID LBPH)
   ┃ ┣ 📜 comfort_model.pkl   # Trọng số mô hình AI: ASHRAE Thermal Comfort
   ┃ ┣ 📜 labels.pkl          # File lưu trữ tên người dùng Face ID
   ┃ ┣ 📜 lbph_model.yml      # Trọng số mô hình AI: Nhận diện khuôn mặt cục bộ
   ┃ ┣ 📜 models.py
   ┃ ┣ 📜 mqtt_client.py      # Module giao tiếp Cloud và thực thi AI Automation
   ┃ ┣ 📜 occupancy_model.pkl # Trọng số mô hình AI: Occupancy Detection
   ┃ ┣ 📜 urls.py             # Định tuyến API và đường dẫn của App
   ┃ ┗ 📜 views.py            # Xử lý logic API, render giao diện Web
   ┣ 📂 smart_home/           # Thư mục cấu hình tổng của Django (settings, wsgi, asgi)
   ┣ 📜 db.sqlite3            # Cơ sở dữ liệu mặc định
   ┗ 📜 manage.py             # Script quản lý và khởi chạy máy chủ
```

## ⚙️ Yêu cầu hệ thống

* **Hệ điều hành:** Windows / macOS / Linux
* **Ngôn ngữ:** Python 3.8 trở lên
* **Phần cứng:** * Laptop có tích hợp Webcam (hoặc Camera rời).
  * Bộ Kit Yolo:Bit kèm mạch mở rộng, cảm biến DHT20, cảm biến ánh sáng, LED và Quạt mini.

## 🚀 Hướng dẫn cài đặt và Khởi chạy

### 1. Thiết lập môi trường ảo
Mở Terminal/Command Prompt tại thư mục gốc của dự án và tạo môi trường ảo để tránh xung đột thư viện:

```bash
python -m venv .venv
```

Kích hoạt môi trường ảo:
* **Trên Windows:** `.\.venv\Scripts\activate`
* **Trên macOS/Linux:** `source .venv/bin/activate`

### 2. Cài đặt thư viện phụ thuộc
Cài đặt các gói thư viện cần thiết cho Backend, AI và Computer Vision:

```bash
python -m pip install django paho-mqtt scikit-learn pandas numpy opencv-contrib-python
```

### 3. Khởi tạo Cơ sở dữ liệu
Đồng bộ các bảng dữ liệu mặc định của Django:

```bash
cd smart_home
python manage.py migrate
```

### 4. Chạy Máy chủ (Gateway)
Khởi động hệ thống. Lúc này, cả luồng Web, luồng quét Camera và kết nối MQTT sẽ chạy song song:

```bash
python manage.py runserver
```

## 🎮 Hướng dẫn sử dụng hệ thống

1. **Truy cập Giao diện:** Mở trình duyệt và truy cập vào địa chỉ `http://127.0.0.1:8000/`.
2. **Đăng nhập:**
   * **Cách 1 (Smart Login):** Cho phép trình duyệt truy cập Camera, nhìn thẳng vào ống kính. Nếu đã đăng ký, hệ thống sẽ nhận diện và tự động hiển thị nút đăng nhập.
   * **Cách 2 (Manual):** Trong quá trình test hoặc nếu chưa đăng ký khuôn mặt, có thể sử dụng mật khẩu dự phòng là `123456`.
3. **Thêm dữ liệu Face ID:** Sử dụng tính năng "Đăng ký" trên màn hình đăng nhập. Thuật toán LBPH sẽ tự động chụp ảnh, huấn luyện (`Online Learning`) và lưu mô hình vào file `lbph_model.yml` ngay lập tức.
4. **Kiểm thử AI:**
   * Thay đổi các thông số môi trường trên bộ Kit Yolo:Bit (che cảm biến ánh sáng, làm nóng cảm biến nhiệt).
   * Dữ liệu sẽ được đẩy lên luồng MQTT, mô hình `.pkl` phân tích và trả về lệnh điều khiển tự động cho Đèn/Quạt, đồng thời cập nhật nút bấm trên Web Dashboard theo thời gian thực.