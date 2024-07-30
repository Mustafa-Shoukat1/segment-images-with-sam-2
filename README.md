# Segment Anything Model 2 (SAM 2) - Project Repository
![image](https://github.com/user-attachments/assets/b16c4745-08c6-424c-9d69-9a2cd0f0d86f)

## Overview

Segment Anything Model 2 (SAM 2) is a powerful foundation model designed for visual segmentation in images and videos. Leveraging an efficient transformer architecture, SAM 2 enables real-time processing and excels across a variety of visual tasks and domains.

This repository provides a comprehensive guide to using SAM 2 for image segmentation, including setup instructions, example usage, and advanced options for fine-tuning the segmentation results.

## Features

- **Versatile Segmentation**: Handles images and videos by treating images as single-frame videos.
- **Efficient Design**: Utilizes a simple transformer architecture with streaming memory for real-time processing.
- **Interactive Data Engine**: Incorporates user interactions to enhance model performance using the extensive SA-V dataset.
- **Robust Performance**: Achieves high accuracy across various tasks and visual domains.
  
![image](https://github.com/user-attachments/assets/16061293-97f7-4fca-8ee1-6c696a6bb2f3)

## Setup

### Prerequisites

- **GPU Access**: Ensure you have access to a GPU. Use the `nvidia-smi` command to check GPU availability. If necessary, configure the notebook settings to use GPU.

```bash
!nvidia-smi
```

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/facebookresearch/segment-anything-2.git
   cd segment-anything-2
   ```

2. **Install Dependencies**

   ```bash
   pip install -e . -q
   pip install -q supervision jupyter_bbox_widget
   ```

3. **Download Checkpoints**

   SAM 2 is available in multiple model sizes. Download the required checkpoints:

   ```bash
   mkdir -p checkpoints
   wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt -P checkpoints
   wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt -P checkpoints
   wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt -P checkpoints
   wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt -P checkpoints
   ```

4. **Download Example Data**

   ```bash
   mkdir -p data
   wget -q https://media.roboflow.com/notebooks/examples/dog.jpeg -P data
   wget -q https://media.roboflow.com/notebooks/examples/dog-2.jpeg -P data
   wget -q https://media.roboflow.com/notebooks/examples/dog-3.jpeg -P data
   wget -q https://media.roboflow.com/notebooks/examples/dog-4.jpeg -P data
   ```

## Usage

### Automated Mask Generation

Generate masks for an image using the `SAM2AutomaticMaskGenerator` class:

```python
import cv2
import torch
import numpy as np
import supervision as sv
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Load model
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = "checkpoints/sam2_hiera_large.pt"
CONFIG = "sam2_hiera_l.yaml"
sam2_model = build_sam2(CONFIG, CHECKPOINT, device=DEVICE, apply_postprocessing=False)

# Generate masks
mask_generator = SAM2AutomaticMaskGenerator(sam2_model)
IMAGE_PATH = "data/dog.jpeg"
image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
sam2_result = mask_generator.generate(image_rgb)

# Visualize results
mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
detections = sv.Detections.from_sam(sam_result=sam2_result)
annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
sv.plot_images_grid(images=[image_bgr, annotated_image], grid_size=(1, 2), titles=['source image', 'segmented image'])
```

### Prompting with Boxes

Use the `SAM2ImagePredictor` class to prompt the model with bounding boxes:

```python
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Initialize predictor
predictor = SAM2ImagePredictor(sam2_model)

# Set image and generate masks with boxes
IMAGE_PATH = "data/dog-2.jpeg"
image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
predictor.set_image(image_rgb)

boxes = np.array([[166, 835, 265, 1010], [472, 885, 640, 1134]])  # Example boxes
masks, scores, logits = predictor.predict(box=boxes)

# Visualize results
box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks=masks), mask=masks.astype(bool))
source_image = box_annotator.annotate(scene=image_bgr.copy(), detections=detections)
segmented_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
sv.plot_images_grid(images=[source_image, segmented_image], grid_size=(1, 2), titles=['source image', 'segmented image'])
```

### Prompting with Points

Generate masks by prompting the model with points:

```python
from jupyter_bbox_widget import BBoxWidget

# Interactive point prompting
def encode_image(filepath):
    with open(filepath, 'rb') as f:
        image_bytes = f.read()
    encoded = str(base64.b64encode(image_bytes), 'utf-8')
    return "data:image/jpg;base64,"+encoded

widget = BBoxWidget()
widget.image = encode_image(IMAGE_PATH)
widget

default_points = [{'x': 330, 'y': 450}, {'x': 191, 'y': 665'}]
points = np.array([[p['x'], p['y']] for p in (widget.bboxes if widget.bboxes else default_points)])
labels = np.ones(points.shape[0])

masks, scores, logits = predictor.predict(point_coords=points, point_labels=labels, multimask_output=True)

# Visualize results
sv.plot_images_grid(images=masks, titles=[f"score: {score:.2f}" for score in scores], grid_size=(1, 3), size=(12, 12))
```

## Advanced Options

Customize the mask generation process with advanced options like `points_per_side`, `pred_iou_thresh`, and more. Refer to the script for details on how to configure these parameters.

## Contributing

Feel free to contribute to this project by opening issues, submitting pull requests, or suggesting improvements. For more details, please refer to the CONTRIBUTING.md file.

## License

This project is licensed under the MIT License. See the LICENSE file for more information.

---
