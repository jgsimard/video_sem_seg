import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptNet(nn.Module):
    def __init__(self):
        pass

    def forward(self, *input):
        pass


class AdaptiveKeyFrameSelector(nn.Module):
    def __init__(self, in_channels=1024):
        self.conv_reduce = nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=3)
        self.conv_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
        self.pred = nn.Linear(256, 1)

    def forward(self, current_frame_low_features, key_frame_low_features):
        cur = F.relu(self.conv_reduce(current_frame_low_features))
        key = F.relu(self.conv_reduce(key_frame_low_features))
        out = F.relu(self.conv_2(cur - key))
        out = F.avg_pool2d(out)
        out = out.view(-1, 256)
        return F.sigmoid(self.pred(out))


class KernelWeightPredictor(nn.Module):
    '''
    Produces the weights used for the Spatially Variant Convolution
    '''
    def __init__(self, in_channels=1024):
        self.conv_reduce = nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(in_channels=256, out_channels=81, kernel_size=1)
        pass

    def forward(self, current_frame_low_features, key_frame_low_features):
        cur = F.relu(self.conv_reduce(current_frame_low_features))
        key = F.relu(self.conv_reduce(key_frame_low_features))
        out = F.relu(self.conv_2(torch.cat((cur, key), 1)))
        return F.softmax(out, 1)  # output dim = N x (9**2) x W x H

class AdaptiveFeaturePropagation(nn.Module):
    def __init__(self, in_channels=1024):
        self.kernel_weight_predictor = KernelWeightPredictor(in_channels)
    def forward(self, current_frame_low_features, key_frame_low_features, key_frame_high_features):
        spatially_variant_kernels = self.kernel_weight_predictor(current_frame_low_features, key_frame_low_features)
        F.conv2d()
        return torch.cat([F.conv2d()], 0)