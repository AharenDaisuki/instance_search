# ROI localizer
import torchvision.models.detection as detection

def localizer_factory(localizer: str):
    if localizer == 'fasterrcnn':
        weights = detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = detection.fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
        return model, weights.transforms()
    elif localizer == 'yolov8n':
        pass
    else:
        raise ValueError(f"Localizer {localizer} is not implemented!")