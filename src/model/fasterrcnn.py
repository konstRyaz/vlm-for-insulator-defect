from __future__ import annotations

from torchvision.models import ResNet50_Weights
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_detector(num_classes: int, pretrained: bool = False):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    weights_backbone = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None

    model = fasterrcnn_resnet50_fpn(weights=weights, weights_backbone=weights_backbone)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
