import sys
sys.path.insert(0, "//home/deepsight2/jg_internship/video_sem_seg/models")
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.modeling.aspp import build_aspp
from models.modeling.decoder import build_decoder
from models.modeling.backbone import build_backbone


class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False, input_channels=3, pretrained=True):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm, input_channels, pretrained=pretrained)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x

    def forward_low(self, input):
        low_level_feat = self.backbone.forward_low(input)
        self.output_shape = input.size()[2:]
        return low_level_feat

    def forward_high(self, x):
        x, low_level_feat = self.backbone.forward_high(x)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        return x

    def interpolate(self, x):
        return F.interpolate(x, size=self.output_shape, mode='bilinear', align_corners=True)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    from torch2trt import torch2trt
    import time
    model = DeepLab(backbone='resnet', output_stride=16)
    model.eval()
    model.cuda()
    input = torch.rand(1, 3, 513, 513)

    print("NORMAL")
    start = time.time()
    for i in range(10):
        output = model(input)
    end = time.time()
    print(f"Time elapsed : {end-start}")
    print(output.size())
    print("TENSOR RT")
    model_trt = torch2trt(model, [input])
    print(model_trt)
    start = time.time()
    for i in range(10):
        output = model(input)
    end = time.time()
    print(f"Time elapsed : {end-start}")


