import paho.mqtt.client as mqtt
import time
import random

BROKER = "broker.hivemq.com"
PORT = 1883

TOPIC_LIGHT = "bk-iot-light"
TOPIC_TEMP = "bk-iot-temp"

client = mqtt.Client()
client.connect(BROKER, PORT, 60)

print("Test started...")
while True:
    light = random.randint(0, 1)
    temp = random.randint(25, 35)

    client.publish(TOPIC_LIGHT, light)
    client.publish(TOPIC_TEMP, temp)

    print(f"Sent light: {light}, temp: {temp}")

    time.sleep(3)