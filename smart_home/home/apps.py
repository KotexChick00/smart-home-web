from django.apps import AppConfig
import os

class HomeConfig(AppConfig):
    name = 'home'

    def ready(self):
        if os.environ.get('RUN_MAIN') != 'true':
            return

        from . import mqtt_client
        mqtt_client.start()