import torch
import os

path = '/Users/vogiaminh/Desktop/Courses/DADN/smart-home-web/module/face_recognition/model/MS1MV3_arcface_r18_p16.pth'
if os.path.exists(path):
    sd = torch.load(path, map_location='cpu')
    if isinstance(sd, dict):
        if 'state_dict' in sd:
            sd = sd['state_dict']
        print(f"Total keys: {len(sd)}")
        for k in sd.keys():
            if k.startswith('layer1.0'):
                print(f"  {k}")
    else:
        print("Not a dict")
else:
    print("File not found.")
