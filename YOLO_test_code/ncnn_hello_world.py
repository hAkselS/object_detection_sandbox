from ultralytics import YOLO 

ncnn_model = YOLO('models/yolov8n_fish_trained_lgds_ncnn_model')

results = ncnn_model.predict(source = 'data/mm_data/00003.png')