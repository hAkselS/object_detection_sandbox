from ultralytics import YOLO
import os 


cwd = os.getcwd()

model_path = os.path.join(cwd, 'models/yolo11n_fish_2016_v1.pt')
# model_path = os.path.join(cwd, 'models/yolo11n_fish_trained_og.pt')
model = YOLO(model_path)



# results = model.predict(source ='/media/gpu_enjoyer/KINGSTON/20240831_172402/20240831.172835.545.001356.jpg')
results = model.predict(source ='/media/gpu_enjoyer/KINGSTON/20240831_172402/20240831.172450.455.000005.jpg')
# results = model.predict(source = 'data/mm_data/00003.png')

for result in results:
    boxes = result.boxes
    result.show()

