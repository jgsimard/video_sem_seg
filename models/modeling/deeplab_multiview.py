import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modeling.deeplab import DeepLab
from models.modeling.merger import build_merger


class DeepLabMultiView(DeepLab):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(DeepLabMultiView, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)
        self.merger = build_merger(num_classes)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input['intensity'])
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        soft_label = F.softmax(x)
        feature = torch.cat(soft_label, input['xyz'][:, 2, :, :])

        score = self.merger(feature, input['xyz'], input['camid'])

        return score


if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())
