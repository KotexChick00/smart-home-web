import json
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.template import loader
from django.views.decorators.csrf import csrf_exempt

from . import mqtt_client
from .mqtt_client import LIGHT_DATA, TEMP_DATA
from .mqtt_client import TOPIC_LED, TOPIC_FAN

def home(request):
    template = loader.get_template('home.html')
    return HttpResponse(template.render())

def get_light_data(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'Invalid request method'}, status=405)
    return JsonResponse(LIGHT_DATA)

def get_temp_data(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'Invalid request method'}, status=405)
    return JsonResponse(TEMP_DATA)

def control_device(request, topic, name):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=405)
    state = json.loads(request.body or '{}').get('state')
    if state == 'ON':
        mqtt_client.publish(topic, '1')
        return JsonResponse({'status': f'{name} turned ON'})
    elif state == 'OFF':
        mqtt_client.publish(topic, '0')
        return JsonResponse({'status': f'{name} turned OFF'})
    return JsonResponse({'error': 'Invalid state'}, status=400)

@csrf_exempt
def control_led(request):
    return control_device(request, TOPIC_LED, "LED")

@csrf_exempt
def control_fan(request):
    return control_device(request, TOPIC_FAN, "Fan")