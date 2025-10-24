# feature extraction
import torch
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

def extractor_factory(backbone: str):
    """ feature extractor factory """
    # vgg
    if backbone == 'vgg16':
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        transforms = models.VGG16_Weights.IMAGENET1K_V1.transforms()
        vgg16_return_nodes = {
            # 'features.4': 'conv1_block',    # After 2nd max pooling
            # 'features.9': 'conv2_block',    # After 3rd max pooling
            # 'features.16': 'conv3_block',   # After 4th max pooling
            'features.23': 'conv4_block',   # After 5th max pooling
            'features.30': 'conv5_block',   # Final conv layer
        }
        model = create_feature_extractor(vgg16, return_nodes=vgg16_return_nodes)
        # Define return nodes for VGG19
    elif backbone == 'vgg19':
        vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        transforms = models.VGG19_Weights.IMAGENET1K_V1.transforms()
        vgg19_return_nodes = {
            # 'features.4': 'conv1_block',    # After 2nd max pooling
            # 'features.9': 'conv2_block',    # After 3rd max pooling
            # 'features.18': 'conv3_block',   # After 4th max pooling
            'features.27': 'conv4_block',   # After 5th max pooling
            'features.36': 'conv5_block',   # Final conv layer
        }        
        model = create_feature_extractor(vgg19, return_nodes=vgg19_return_nodes)
    # efficientnet
    elif backbone == 'efficientnet_v2_s':
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        transforms = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms()
    elif backbone == 'efficientnet_v2_m':
        model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        transforms = models.EfficientNet_V2_M_Weights.IMAGENET1K_V1.transforms()
    elif backbone == 'efficientnet_v2_l': 
        model = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        transforms = models.EfficientNet_V2_L_Weights.IMAGENET1K_V1.transforms()
    # resnet
    elif backbone == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        transforms = models.ResNet50_Weights.IMAGENET1K_V2.transforms()
    elif backbone == 'resnet101':
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        transforms = models.ResNet101_Weights.IMAGENET1K_V2.transforms()
    elif backbone == 'resnet152': # OK
        model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        transforms = models.ResNet152_Weights.IMAGENET1K_V2.transforms()
    # vit
    elif backbone == 'vit_b_16': 
        vit_b_16 = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        transforms = models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()
        vit_b_16_return_nodes = {
            # 'encoder.ln': 'pre_classifier_norm',          # Layer norm before classifier
            # 'encoder.layers.5': 'mid_transformer_block',  # Middle transformer block
            'encoder.layers.encoder_layer_11': 'final_transformer_block', # Final transformer block
            # 'heads.head': 'classifier'                    # Classification head
        }
        # train_nodes, eval_nodes = get_graph_node_names(vit_b_16)
        model = create_feature_extractor(vit_b_16, return_nodes=vit_b_16_return_nodes)
    elif backbone == 'vit_l_16': 
        vit_l_16 = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        transforms = models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()
        vit_l_16_return_nodes = {
            # 'encoder.ln': 'pre_classifier_norm',
            # 'encoder.layers.11': 'quarter_transformer_block',
            'encoder.layers.encoder_layer_23': 'final_transformer_block',
            # 'heads.head': 'classifier'
        }
        # train_nodes, eval_nodes = get_graph_node_names(vit_l_16)
        model = create_feature_extractor(vit_l_16, return_nodes=vit_l_16_return_nodes)
    else: 
        raise ValueError(f"Backbone {backbone} is not implemented!")
    return model, transforms
