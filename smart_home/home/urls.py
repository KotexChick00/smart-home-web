from django.urls import path
from  . import views

urlpatterns = [
    path('', views.login_view, name='login'),
    path('dashboard/', views.home, name='home'),
    path('logout/', views.logout_view, name='logout'),
    path('register/', views.register_view, name='register_view'),
    path('manage_users/', views.manage_users_view, name='manage_users_view'),
    path('delete_user/', views.delete_user, name='delete_user'),
    path('confirm_login/', views.confirm_login, name='confirm_login'),
    path('check_face/', views.check_face, name='check_face'),
    path('register_face/', views.register_face, name='register_face'),
    path('set_camera/', views.set_camera, name='set_camera'),
    path('verify_password/', views.verify_password, name='verify_password'),
    path('get_light_data/', views.get_light_data),
    path('get_temp_data/', views.get_temp_data),
    path('get_humi_data/', views.get_humi_data),
    path('get_ir_data/', views.get_ir_data),
    path('control_led/', views.control_led),
    path('control_fan/', views.control_fan),
    path('video_feed/', views.video_feed),
    path('get_device_status/', views.get_device_status, name='get_device_status'),
    path('toggle_ai/', views.toggle_ai, name='toggle_ai'),
    path('toggle_camera/', views.toggle_camera, name='toggle_camera'),
]