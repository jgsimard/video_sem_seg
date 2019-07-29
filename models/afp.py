import torch
import torch.multiprocessing  # TO DO
import torch.nn as nn
import torch.nn.functional as F
import math
from models.modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

from models.rep_flow_layer import FlowLayer


# from opt_einsum import contract


def _init_weight(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, SynchronizedBatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

class AdaptNet(nn.Module):
    def __init__(self, in_channels=128, size=[513, 513]):
        super(AdaptNet, self).__init__()
        self.apapt_low = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # self.apapt_high = nn.Sequential(
        #     nn.Conv2d(in_channels=11, out_channels=256, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        #     nn.ReLU()
        # )
        self.mix = nn.Sequential(
            nn.Conv2d(in_channels=256+11, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=11, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.upsample = nn.Upsample(size=size, mode='bilinear', align_corners=True)
        _init_weight(self)

    def forward(self, low, high):
        adapted_low = self.apapt_low(low)
        # adapted_high = self.apapt_high(high)
        adapted_high = high
        x = torch.cat((adapted_low, adapted_high), dim=1)
        x = self.mix(x)
        x = self.upsample(x)
        return x

class AdaptiveKeyFrameSelector(nn.Module):
    def __init__(self, in_channels=128):
        super(AdaptiveKeyFrameSelector, self).__init__()
        self.conv_reduce = nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=3)
        self.conv_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
        self.pred = nn.Linear(256, 1)
        _init_weight(self)

    def forward(self, current_frame_low_features, key_frame_low_features):
        cur = F.relu(self.conv_reduce(current_frame_low_features))
        key = F.relu(self.conv_reduce(key_frame_low_features))
        out = F.relu(self.conv_2(cur - key))
        b, c, h, w = out.shape
        out = F.avg_pool2d(out, kernel_size=(h, w))
        out = out.view(-1, 256)
        return F.sigmoid(self.pred(out))


class KernelWeightPredictor(nn.Module):
    # Produces the weights used for the Spatially Variant Convolution
    def __init__(self, in_channels=128, kernel_size=7):
        self.k = kernel_size
        super(KernelWeightPredictor, self).__init__()
        self.conv_reduce = nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(in_channels=256, out_channels=self.k ** 2, kernel_size=1)
        _init_weight(self)

    def forward(self, current_frame_low_features, key_frame_low_features):
        # compute spatially variant kernels
        cur = F.relu(self.conv_reduce(current_frame_low_features))
        key = F.relu(self.conv_reduce(key_frame_low_features))
        x = F.relu(self.conv_2(torch.cat((cur, key), 1)))
        x = F.relu(self.conv_3(x))
        x = F.softmax(x, 1)  # output dim = N x (K**2) x H x W
        b, k2, h, w = x.shape
        x = x.permute(0,2,3,1).view(b, h, w,self.k, self.k)
        #
        # x = x.view(b, self.k, self.k, h, w)
        return x


class KernelWeightPredictorFlow(nn.Module):
    # Produces the weights used for the Spatially Variant Convolution using a representation flow layer
    def __init__(self, in_channels=128, kernel_size=7, learnable=[1,1,1,1,1], flow_channels=32, n_iter=5):
        self.k = kernel_size
        super(KernelWeightPredictorFlow, self).__init__()
        self.conv_reduce = nn.Conv2d(in_channels=in_channels, out_channels=flow_channels, kernel_size=3, padding=1)
        self.flow_layer = FlowLayer(channels=flow_channels, n_iter=n_iter, params=learnable)
        self.conv_expand = nn.Conv2d(in_channels=flow_channels, out_channels=self.k ** 2, kernel_size=3, padding=1)

    def forward(self, current_frame_low_features, key_frame_low_features):
        cur = F.relu(self.conv_reduce(current_frame_low_features))
        key = F.relu(self.conv_reduce(key_frame_low_features))
        x = torch.stack((key, cur), 2) # b, c, t, h, w = x.size()
        x = self.flow_layer(x)
        x = x.squeeze()
        x = F.relu(self.conv_expand(x))
        b, k2, h, w = x.shape
        x = x.permute(0, 2, 3, 1).view(b, h, w, self.k, self.k)
        return x


class SpatiallyVariantConvolution(nn.Module):
    def __init__(self, kernel_size):
        super(SpatiallyVariantConvolution, self).__init__()
        self.pad = nn.ZeroPad2d(kernel_size // 2)
        self.ks = kernel_size

    def forward(self, kernels, features):
        ks = self.ks
        bs, in_c, h_, w_ = features.size()
        features = self.pad(features)
        bs, in_c, h, w = features.size()
        strided_features = features.as_strided((bs, in_c, h - ks + 1, w - ks + 1, ks, ks),
                                               (h * w * in_c, h * w, w, 1, w, 1))
        # out = torch.matmul(strided_features.view(bs, in_c, h_*w_, ks*ks), kernels.view(bs, ks*ks, h_*w_))
        # out = contract('bchwkl,bklhw->bchw', strided_features, kernels, backend='torch')
        out = torch.einsum('bchwkl,bhwkl->bchw', [strided_features, kernels])
        return out


class AdaptiveFeaturePropagation(nn.Module):
    def __init__(self, in_channels=128, kernel_size=7, flow=False):
        super(AdaptiveFeaturePropagation, self).__init__()
        # self.upsample = nn.Upsample(size=size, mode='bilinear')
        if flow:
            self.kernel_weight_predictor = KernelWeightPredictorFlow(in_channels, kernel_size)
        else:
            self.kernel_weight_predictor = KernelWeightPredictor(in_channels, kernel_size=kernel_size)
        self.spatially_variant_convolution = SpatiallyVariantConvolution(kernel_size=kernel_size)
        _init_weight(self)

    def forward(self, current_frame_low_features, key_frame_low_features, key_frame_high_features):
        # current_frame_low_features_upsampled = self.upsample(current_frame_low_features)
        # key_frame_low_features_upsampled = self.upsample(key_frame_low_features)
        spatially_variant_kernels = self.kernel_weight_predictor(current_frame_low_features,
                                                                 key_frame_low_features)
        return self.spatially_variant_convolution(spatially_variant_kernels, key_frame_high_features)


class LowLatencyModel(nn.Module):
    def __init__(self, spatial_model, threshold=0.3, fixed_schedule=5, kernel_size=11, flow=False):
        super(LowLatencyModel, self).__init__()
        self.adaptive_feature_propagation = AdaptiveFeaturePropagation(in_channels=128, kernel_size=kernel_size, flow=flow)
        self.adaptive_key_frame_selector = AdaptiveKeyFrameSelector()
        self.adapt_net = AdaptNet()
        self.spatial_model = spatial_model

        self.key_frame_low_features = None
        self.key_frame_high_features = None
        self.cur_frame_low_features = None
        self.steps_same_key_frame = 0
        self.fixed_schedule = fixed_schedule
        self.threshold = threshold
        _init_weight(self)

    def forward_spatial_model(self, cur_frame_low_features):
        self.key_frame_low_features = cur_frame_low_features
        self.key_frame_high_features = self.spatial_model.forward_high(cur_frame_low_features)
        return self.key_frame_high_features

    def compute_deviation(self, features_1, features_2):
        seg_map_1 = torch.argmax(features_1, dim=1)
        seg_map_2 = torch.argmax(features_2, dim=1)
        b, c, h, w = features_1.shape
        return torch.einsum('bhw->b', [torch.eq(seg_map_1, seg_map_2)]) / (h * w)

    def forward(self, input, random_input=None, train=False):
        # input is new frame on which to do inference
        # random_input is the past key frame
        if train:
            # input is new frame on which to do inference
            # random_input is the past key frame
            random_frame_low_features = self.spatial_model.forward_low(random_input)
            self.forward_spatial_model(random_frame_low_features)
            cur_frame_low_features = self.spatial_model.forward_low(input)

            # deviation
            deviation = self.adaptive_key_frame_selector(cur_frame_low_features, self.key_frame_low_features)
            cur_frame_high_features = self.spatial_model.forward_high(cur_frame_low_features)
            real_deviation = self.compute_deviation(cur_frame_high_features, self.key_frame_high_features)

            # feature propagation
            x = self.adaptive_feature_propagation(cur_frame_low_features, self.key_frame_low_features,
                                                  self.key_frame_high_features)
            x = self.adapt_net(cur_frame_low_features, x)

            return x, deviation, real_deviation

        else:
            cur_frame_low_features = self.spatial_model.forward_low(input)
            if self.key_frame_low_features is None:
                return self.forward_spatial_model(cur_frame_low_features)

            new_key_frame = False
            if self.fixed_schedule is not None:
                if self.steps_same_key_frame >= self.fixed_schedule:
                    new_key_frame = True
            else:
                deviation = self.adaptive_key_frame_selector(cur_frame_low_features, self.key_frame_low_features)
                if deviation > self.threshold:
                    new_key_frame = True

            if new_key_frame:
                return self.forward_spatial_model(cur_frame_low_features)
            else:
                x = self.adaptive_feature_propagation(cur_frame_low_features, self.key_frame_low_features,
                                                      self.key_frame_high_features)
                x = self.adapt_net(x, cur_frame_low_features)
                return x


if __name__ == '__main__':
    import time

    # b,c,h,w = 2,11,600,600
    # k=9
    b, c, h, w = 4, 11, 129, 129
    k = 7
    N = 100
    features = torch.rand((b, c, h, w)).cuda()
    kernels = torch.rand((b, k, k, h, w)).cuda()

    svc = SpatiallyVariantConvolution(k)
    svc(kernels, features)
    start = time.time()
    for i in range(N):
        out_1 = svc(kernels, features)
    print(f"unfold={time.time() - start}, memory = {torch.cuda.memory_allocated()}")

    # svc_2 = SpatiallyVariantConvolutionStridded(k)
    # svc_2(kernels, features)
    # start = time.time()
    # for i in range(N):
    #     out_2 = svc_2(kernels, features)
    # print(f"stridded={time.time() - start}, memory = {torch.cuda.memory_allocated()}")

    print(torch.equal(out_1, out_2))
