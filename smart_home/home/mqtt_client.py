import paho.mqtt.client as mqtt
import joblib
import numpy as np
import pandas as pd
import os

BROKER = 'broker.hivemq.com'
PORT = 1883

TOPIC_LIGHT = "tnaker-bk-iot-light"
TOPIC_TEMP = "tnaker-bk-iot-temp"
TOPIC_HUMI = "tnaker-bk-iot-humi"
TOPIC_LED = "tnaker-bk-iot-led"
TOPIC_FAN = "tnaker-bk-iot-fan"
TOPIC_IR = "tnaker-bk-iot-ir"

# Các biến toàn cục lưu trữ dữ liệu cho Web
LIGHT_DATA = {'light': 0}
TEMP_DATA = {'temp': 25.0}
HUMI_DATA = {'humi': 50.0}
IR_DATA = {'ir': 0}

# Lưu trạng thái hiện tại của thiết bị để đồng bộ với nút bấm trên Web
DEVICE_STATE = {'led': '0', 'fan': '0'} 

AI_MODE = True
IS_LOGGED_IN = False
LAST_ACTIVE_TIME = 0

client = mqtt.Client()
model_occupancy = None
model_comfort = None

def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT Broker...")
    client.subscribe(TOPIC_TEMP)
    client.subscribe(TOPIC_HUMI)
    client.subscribe(TOPIC_LIGHT)
    client.subscribe(TOPIC_IR)
    client.subscribe(TOPIC_LED)
    client.subscribe(TOPIC_FAN)

def on_message(client, userdata, msg):
    global model_occupancy, model_comfort, AI_MODE, IS_LOGGED_IN, LAST_ACTIVE_TIME
    topic = msg.topic
    payload = msg.payload.decode()

    print(f"[MQTT] Nhận {payload} từ {topic}")

    # 1. Cập nhật trạng thái công tắc (Do Web bấm hoặc do AI tự kích hoạt)
    if topic == TOPIC_LED:
        DEVICE_STATE['led'] = payload
    elif topic == TOPIC_FAN:
        DEVICE_STATE['fan'] = payload

    # 2. Cập nhật dữ liệu cảm biến
    try:
        val = float(payload)
        if topic == TOPIC_TEMP:
            TEMP_DATA['temp'] = val
        elif topic == TOPIC_HUMI:
            HUMI_DATA['humi'] = val
        elif topic == TOPIC_LIGHT:
            LIGHT_DATA['light'] = val
        elif topic == TOPIC_IR:
            IR_DATA['ir'] = val
    except ValueError:
        pass 

    import time
    # Kiểm tra xem có người dùng đang đăng nhập và đang xem Dashboard không
    is_active = IS_LOGGED_IN and (time.time() - LAST_ACTIVE_TIME < 6.0)

    if not is_active:
        # Nếu chưa đăng nhập hoặc đóng tab, tắt tất cả thiết bị
        if DEVICE_STATE['led'] != '0':
            publish(TOPIC_LED, '0')
            DEVICE_STATE['led'] = '0'
        if DEVICE_STATE['fan'] != '0':
            publish(TOPIC_FAN, '0')
            DEVICE_STATE['fan'] = '0'
    else:
        # 3. 🤖 THỰC THI TRÍ TUỆ NHÂN TẠO (chỉ chạy nếu AI_MODE được bật)
        if AI_MODE:
            # AI 1: Bật/Tắt ĐÈN dựa trên mô hình Occupancy Detection
            if model_occupancy and topic in [TOPIC_TEMP, TOPIC_HUMI, TOPIC_LIGHT]:
                ai_has_person = 1 # Force always occupied (chỉnh thành luôn có người)
                
                if ai_has_person == 1:
                    publish(TOPIC_LED, "1")
                else:
                    publish(TOPIC_LED, "0")

            # AI 2: Bật/Tắt QUẠT dựa trên mô hình ASHRAE Thermal Comfort
            if model_comfort and topic in [TOPIC_TEMP, TOPIC_HUMI]:
                input_comf = pd.DataFrame(
                    [[TEMP_DATA['temp'], HUMI_DATA['humi']]],
                    columns=['Air temperature (C)', 'Relative humidity (%)']
                )
                predicted_pmv = model_comfort.predict(input_comf)[0]
                
                if predicted_pmv > 0.5:
                    publish(TOPIC_FAN, "1")
                else:
                    publish(TOPIC_FAN, "0")

client.on_connect = on_connect
client.on_message = on_message

def start():
    """Hàm khởi tạo được gọi từ apps.py khi server Django bắt đầu chạy"""
    global model_occupancy, model_comfort
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    try:
        model_occupancy = joblib.load(os.path.join(BASE_DIR, 'occupancy_model.pkl'))
        model_comfort = joblib.load(os.path.join(BASE_DIR, 'comfort_model.pkl'))
        print("--- 🤖 MQTT Gateway: Đã nạp thành công các mô hình AI ---")
    except Exception as e:
        print(f"--- ⚠️ Lỗi nạp AI: {e}. Hệ thống Web vẫn hoạt động bình thường... ---")

    client.connect(BROKER, PORT, 60)
    client.loop_start()

def publish(topic, value):
    """Hàm xuất API để views.py hoặc AI gọi khi cần đổi trạng thái thiết bị"""
    client.publish(topic, value)