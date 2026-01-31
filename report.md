# Report

## Group Members
- Ferrandez Naoki (ferrandez@et.esiea.fr)
- Masi Lucca (masi@et.esiea.fr)

---

## Dataset

**Name & source**  
The dataset is a **person-only subset of the COCO 2017 dataset**, derived from the official COCO annotations:  
https://cocodataset.org

Only images containing the `person` class were retained, along with their polygon-based segmentation masks.

**Preprocessing & splits**
- Training set: ~2000 images  
- Validation set: ~200 images  
- Images resized to **416 × 416**
- Binary segmentation task: *person vs background*
- Normalization and augmentation handled internally by the YOLOv8 pipeline

---

## Model

**Architecture**
- Final model: **YOLOv8n-seg**
- Task: instance segmentation
- Pretrained on COCO, then fine-tuned on the person subset

An initial attempt was made to train a **UNet-style segmentation model from scratch** (still present in the codebase). However, this approach proved too computationally expensive and time-consuming for the project constraints. For feasibility reasons, we pivoted to fine-tuning a pretrained YOLO model.

The YOLO model training and pipeline can be found in the **yolo8seg.ipynb** file.

**Key hyperparameters**
- Image size: 416
- Batch size: 8
- Epochs: 30
- Optimizer: AdamW (auto-selected)
- Device: Apple Silicon MPS

**Loss & optimization**
- YOLOv8 multi-task loss (box, classification, segmentation, DFL)
- Optimizer: AdamW
- Learning rate automatically tuned by the framework

---

## Results

**Quantitative metrics (validation set)**

| Metric | Bounding Box | Segmentation Mask |
|------|-------------|------------------|
| Precision | 0.826 | 0.813 |
| Recall | 0.210 | 0.206 |
| mAP@50 | 0.521 | 0.513 |
| mAP@50–95 | 0.357 | 0.312 |

While the model does not achieve the highest possible mIoU, it produces **mostly consistent and visually coherent person segmentation masks** across a wide range of scenes.

**Example outputs**
- Demo images captured during live testing (input + segmentation mask)

- Mask with picture
<p float="left">
  <img src="/picture_results/BB79BEF0-C45A-4E27-AB18-F34E512E925F.jpeg" width="45%" />
</p>

- Mask with picture and blur
<p float="left">
  <img src="/picture_results/79D99B4E-AC2A-4880-A6ED-D987CE3B9813.jpeg" width="45%" />
</p>

- Validation set predictions:

- Ground truth labels  
<p float="left">
  <img src="/runs/segment/val5/val_batch1_labels.jpg" width="45%" />
</p>


- Model predictions  
<p float="left">
  <img src="/runs/segment/val5/val_batch1_pred.jpg " width="45%" />
</p>


We can see that the model sees most of the people with some errors.

---

## Lessons Learned

1. **Training segmentation models from scratch is costly**  
   Even for binary segmentation, models like UNet require significant tuning, data, and compute to reach competitive performance.

2. **Pretrained models are essential under time constraints**  
   Fine-tuning a pretrained YOLOv8 segmentation model enabled fast convergence and reliable results within the project deadline.

3. **Accuracy vs feasibility is a real trade-off**  
   Although the model does not achieve the best possible mIoU, it performs well globally and meets the functional objectives of the project. Further training with more epochs and compute would likely improve performance.
