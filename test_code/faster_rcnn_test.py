from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image, convert_image_dtype

# Step 1: Read the image
img = read_image("detr_test_code/LeFish.png")

# Convert the image to 3 channels (RGB) if it has 4 (RGBA)
if img.shape[0] == 4:
    img = img[:3, :, :]  # Keep only the first 3 channels (RGB)

# Step 2: Initialize model with the best available weights
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.8)
model.eval()

# Step 3: Initialize the inference transforms
preprocess = weights.transforms()

# Step 4: Apply inference preprocessing transforms
batch = [preprocess(img)]

# Step 5: Use the model and visualize the prediction
prediction = model(batch)[0]
labels = [weights.meta["categories"][i] for i in prediction["labels"]]
box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                          labels=labels,
                          colors="red",
                          width=4)

im = to_pil_image(box.detach())
im.show()
