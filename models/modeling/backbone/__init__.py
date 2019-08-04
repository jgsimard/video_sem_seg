from models.modeling.backbone import resnet, xception, drn, mobilenet


def build_backbone(backbone, output_stride, BatchNorm, input_channels, pretrained=True):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm, input_channels)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm, pretrained, input_channels)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm, input_channels)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm, input_channels)
    else:
        raise NotImplementedError
