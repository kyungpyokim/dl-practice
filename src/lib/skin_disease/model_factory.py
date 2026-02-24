import torch.nn as nn
from torchvision import models


def get_model_and_preprocess(model_name, num_classes=5):
    if model_name == 'resnet34':
        weights = models.ResNet34_Weights.DEFAULT
        model = models.resnet34(weights=weights)

        for param in model.parameters():
            param.requires_grad = False

        model.fc = nn.Linear(model.fc.in_features, num_classes)

        for param in model.layer4.parameters():
            param.requires_grad = True

    elif model_name == 'convnext_tiny':
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        model = models.convnext_tiny(weights=weights)

        for param in model.parameters():
            param.requires_grad = False

        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

        for param in model.classifier.parameters():
            param.requires_grad = True
        for param in model.stages[3].parameters():
            param.requires_grad = True
    elif model_name == 'efficientnet_v2_s':
        weights = models.EfficientNet_V2_S_Weights.DEFAULT
        model = models.efficientnet_v2_s(weights=weights)

        for param in model.parameters():
            param.requires_grad = False

        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

        for param in model.features[7].parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f'Unknown model: {model_name}')

    return model, weights.transforms()
