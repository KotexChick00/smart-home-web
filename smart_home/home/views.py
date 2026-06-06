import json
import time
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
from django.template import loader
from django.views.decorators.csrf import csrf_exempt

from . import mqtt_client
from .mqtt_client import LIGHT_DATA, TEMP_DATA, HUMI_DATA, IR_DATA
from .mqtt_client import TOPIC_LED, TOPIC_FAN

def login_view(request):
    if request.session.get('authenticated'):
        return redirect('home')
    # Tắt thiết bị khi chưa đăng nhập
    mqtt_client.IS_LOGGED_IN = False
    mqtt_client.publish(mqtt_client.TOPIC_LED, '0')
    mqtt_client.publish(mqtt_client.TOPIC_FAN, '0')
    mqtt_client.DEVICE_STATE['led'] = '0'
    mqtt_client.DEVICE_STATE['fan'] = '0'
    return render(request, 'login.html')

def home(request):
    if not request.session.get('authenticated'):
        return redirect('login')
    # Đánh dấu người dùng đã đăng nhập và đang hoạt động
    mqtt_client.IS_LOGGED_IN = True
    mqtt_client.LAST_ACTIVE_TIME = time.time()
    return render(request, 'home.html')

def logout_view(request):
    request.session.flush()
    # Tắt thiết bị khi đăng xuất
    mqtt_client.IS_LOGGED_IN = False
    mqtt_client.publish(mqtt_client.TOPIC_LED, '0')
    mqtt_client.publish(mqtt_client.TOPIC_FAN, '0')
    mqtt_client.DEVICE_STATE['led'] = '0'
    mqtt_client.DEVICE_STATE['fan'] = '0'
    return redirect('login')

def register_view(request):
    return render(request, 'register.html')

def manage_users_view(request):
    # Trang quản lý danh sách user
    from . import camera
    users = camera.label_dict.items()
    return render(request, 'manage_users.html', {'users': users})

@csrf_exempt
def delete_user(request):
    if request.method == 'POST':
        data = json.loads(request.body or '{}')
        index = data.get('index')
        if index is not None:
            from . import camera
            success = camera.delete_user(int(index))
            if success:
                return JsonResponse({'status': 'success'})
    return JsonResponse({'status': 'error'})

@csrf_exempt
def confirm_login(request):
    if request.method == 'POST':
        data = json.loads(request.body or '{}')
        user_name = data.get('user_name')
        if user_name:
            request.session['authenticated'] = True
            request.session['user_name'] = user_name
            return JsonResponse({'status': 'success'})
    return JsonResponse({'status': 'error'})

@csrf_exempt
def verify_password(request):
    if request.method == 'POST':
        data = json.loads(request.body or '{}')
        password = data.get('password')
        if password == '123456':
            request.session['authenticated'] = True
            return JsonResponse({'status': 'success'})
        return JsonResponse({'status': 'error', 'message': 'Sai mật khẩu'})
    return JsonResponse({'error': 'Invalid request method'}, status=405)

def check_face(request):
    try:
        from . import camera
        last_time, user_name = camera.get_recognition_status()
        current_time = time.time()
        
        # Nếu nhận diện được bất kỳ user nào trong database trong vòng 5 giây gần nhất
        if current_time - last_time < 5.0 and user_name is not None:
            return JsonResponse({
                'status': 'success', 
                'user': user_name
            })
            
        return JsonResponse({'status': 'waiting'})
    except Exception as e:
        print(f"Error in check_face: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    
@csrf_exempt
def register_face(request):
    if request.method == 'POST':
        data = json.loads(request.body or '{}')
        name = data.get('name')
        if not name:
            return JsonResponse({'status': 'error', 'message': 'Vui lòng nhập tên'})
            
        from . import camera
        success, message = camera.register_new_user(name)
        
        if success:
            return JsonResponse({'status': 'success', 'message': message})
        else:
            return JsonResponse({'status': 'error', 'message': message})
            
    return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
def set_camera(request):
    if request.method == 'POST':
        data = json.loads(request.body or '{}')
        idx = data.get('camera_index', 0)
        from . import camera
        camera.set_camera_index(idx)
        return JsonResponse({'status': 'success', 'camera_index': idx})
    return JsonResponse({'error': 'Invalid request method'}, status=405)

def get_light_data(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'Invalid request method'}, status=405)
    return JsonResponse(LIGHT_DATA)

def get_temp_data(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'Invalid request method'}, status=405)
    return JsonResponse(TEMP_DATA)

def get_humi_data(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'Invalid request method'}, status=405)
    return JsonResponse(HUMI_DATA)

def get_ir_data(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'Invalid request method'}, status=405)
    return JsonResponse(IR_DATA)

def control_device(request, topic, name):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=405)
    state = json.loads(request.body or '{}').get('state')
    key = name.lower()
    if state == 'ON':
        mqtt_client.publish(topic, '1')
        mqtt_client.DEVICE_STATE[key] = '1'
        return JsonResponse({'status': f'{name} turned ON'})
    elif state == 'OFF':
        mqtt_client.publish(topic, '0')
        mqtt_client.DEVICE_STATE[key] = '0'
        return JsonResponse({'status': f'{name} turned OFF'})
    return JsonResponse({'error': 'Invalid state'}, status=400)

@csrf_exempt
def control_led(request):
    return control_device(request, TOPIC_LED, "LED")

@csrf_exempt
def control_fan(request):
    return control_device(request, TOPIC_FAN, "Fan")

def video_feed(request):
    from . import camera
    mode = request.GET.get('mode', 'recognition')
    return StreamingHttpResponse(camera.gen_frames(mode), content_type='multipart/x-mixed-replace; boundary=frame')

from django.http import JsonResponse
from . import mqtt_client 
from django.views.decorators.csrf import csrf_exempt
import json

def get_device_status(request):
    """API trả về trạng thái hiện tại của Đèn, Quạt và Chế độ AI cho giao diện Web"""
    if request.session.get('authenticated'):
        mqtt_client.IS_LOGGED_IN = True
        mqtt_client.LAST_ACTIVE_TIME = time.time()
    else:
        mqtt_client.IS_LOGGED_IN = False

    from . import camera
    return JsonResponse({
        'led': mqtt_client.DEVICE_STATE.get('led', '0'),
        'fan': mqtt_client.DEVICE_STATE.get('fan', '0'),
        'ai_mode': mqtt_client.AI_MODE,
        'camera_running': camera.streamer.running
    })

@csrf_exempt
def toggle_ai(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body or '{}')
            state = data.get('state') # "ON" or "OFF"
            if state == "ON":
                mqtt_client.AI_MODE = True
            elif state == "OFF":
                mqtt_client.AI_MODE = False
            return JsonResponse({'status': 'success', 'ai_mode': mqtt_client.AI_MODE})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
def toggle_camera(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body or '{}')
            state = data.get('state') # "ON" or "OFF"
            from . import camera
            if state == "ON":
                camera.streamer.start()
            elif state == "OFF":
                camera.streamer.stop()
            return JsonResponse({'status': 'success', 'camera_running': camera.streamer.running})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    return JsonResponse({'error': 'Invalid request method'}, status=405)