from django.urls import path
from  . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('get_light_data/', views.get_light_data),
    path('get_temp_data/', views.get_temp_data),
    path('get_humi_data/', views.get_humi_data),
    path('get_ir_data/', views.get_ir_data),
    path('control_led/', views.control_led),
    path('control_fan/', views.control_fan),
    path('video_feed/', views.video_feed),
]