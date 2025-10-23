# feature extraction
import torch
import torchvision.models as models

def extractor_factory(backbone: str):
    """ feature extractor factory """
    # model factory
    if backbone == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    elif backbone == 'vgg19':
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    elif backbone == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    elif backbone == 'resnet101':
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
    elif backbone == 'resnet152':
        model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
    else: 
        raise ValueError(f"Backbone {backbone} is not implemented!")
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    return model
