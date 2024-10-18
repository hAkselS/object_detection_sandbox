from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO('models/yolov8n_fish_trained_lgds.pt')

# Export the model to NCNN format
# Int 8 makes the model less accurate but faster
model.export(format="ncnn", half=False, imgsz=224)  # creates '/yolo11n_ncnn_model'

# Load the exported NCNN model
# ncnn_model = YOLO("./yolo11n_ncnn_model")

# # Run inference
# results = ncnn_model("https://ultralytics.com/images/bus.jpg")