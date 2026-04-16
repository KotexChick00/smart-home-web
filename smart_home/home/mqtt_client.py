import paho.mqtt.client as mqtt

BROKER = 'broker.hivemq.com'
PORT = 1883

TOPIC_LIGHT = "bk-iot-light"
TOPIC_TEMP = "bk-iot-temp"
TOPIC_HUMI = "bk-iot-humi"
TOPIC_LED = "bk-iot-led"
TOPIC_FAN = "bk-iot-fan"
TOPIC_IR = "bk-iot-ir"

LIGHT_DATA = {'value': None}
TEMP_DATA = {'temp': None}
HUMI_DATA = {'humi': None}
IR_DATA = {'code': None}

def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT Broker...")

    client.subscribe(TOPIC_LIGHT)
    client.subscribe(TOPIC_TEMP)
    client.subscribe(TOPIC_HUMI)
    client.subscribe(TOPIC_IR)

def on_message(client, userdata, msg):
    global LIGHT_DATA, TEMP_DATA, HUMI_DATA, IR_DATA
    topic = msg.topic
    payload = msg.payload.decode()

    if topic == TOPIC_LIGHT:
        LIGHT_DATA['value'] = payload
    elif topic == TOPIC_TEMP:
        TEMP_DATA['temp'] = payload
    elif topic == TOPIC_HUMI:
        HUMI_DATA['humi'] = payload
    elif topic == TOPIC_IR:
        IR_DATA['code'] = payload

    print(f"Received {payload} from {topic}")

def publish(topic, message):
    client.publish(topic, message)

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

def start():
    client.connect(BROKER, PORT, 60)
    client.loop_start()