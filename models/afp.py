import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing # TO DO

from models.rep_flow_layer import FlowLayer



class AdaptNet(nn.Module):
    def __init__(self):
        super(AdaptNet, self).__init__()
        pass

    def forward(self, *input):
        pass


class AdaptiveKeyFrameSelector(nn.Module):
    def __init__(self, in_channels=1024):
        super(AdaptiveKeyFrameSelector, self).__init__()
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
    def __init__(self, in_channels=1024, k = 9):
        self.k = k
        super(KernelWeightPredictor, self).__init__()
        self.conv_reduce = nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(in_channels=256, out_channels=81, kernel_size=1)
        pass

    def forward(self, current_frame_low_features, key_frame_low_features):
        #compute spatially variant kernels
        cur = F.relu(self.conv_reduce(current_frame_low_features))
        key = F.relu(self.conv_reduce(key_frame_low_features))
        x = F.relu(self.conv_2(torch.cat((cur, key), 1)))
        x = F.relu(self.conv_3(x))
        x = F.softmax(x, 1)  # output dim = N x (K**2) x H x W
        b, k2, h, w = x.shape
        x = x.view(b, self.k, self.k, h, w)
        return x

class KernelWeightPredictorFlow(nn.Module):
    '''
    Produces the weights used for the Spatially Variant Convolution
    '''
    def __init__(self, in_channels=1024, k = 9, learnable=True, flow_channels=32, n_iter=5):
        self.k = k
        super(KernelWeightPredictorFlow, self).__init__()
        self.conv_reduce = nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=3, padding=1)
        self.flow_layer = FlowLayer(channels=flow_channels, n_iter=n_iter, params=learnable)
        self.conv_expand = nn.Conv2d(in_channels=flow_channels, out_channels=in_channels, kernel_size=3, padding=1)

    def forward(self, current_frame_low_features, key_frame_low_features):
        cur = F.relu(self.conv_reduce(current_frame_low_features))
        key = F.relu(self.conv_reduce(key_frame_low_features))
        x = torch.stack((key, cur), 2)
        b, c, t, h, w = x.size()
        u, v = self.flow_layer(x[:, :, :-1].permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w),
                               x[:, :, 1:].permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w))
        x = torch.cat([u, v], dim=1)
        x = x.view(b, t - 1, c * 2, h, w).permute(0, 2, 1, 3, 4).contiguous().squeeze()  # shape BxC*2xHxW
        x = F.relu(self.conv_expand(x))
        return x

class SpatiallyVariantConvolution(nn.Module):
    def __init__(self, kernel_size):
        super(SpatiallyVariantConvolution, self).__init__()
        self.unfold = nn.Unfold(kernel_size=(kernel_size, kernel_size), padding=kernel_size // 2)

    def forward(self, kernels, features):
        assert len(kernels.shape) == 5, 'need the shape b x k x k x h x w'
        assert len(features.shape) == 4, 'need the shape b x c x h x w'
        features = self.pad(features)
        b, c, h, w = features.shape
        _, _, _, k, _ = kernels.shape
        features_unfold = self.unfold(features).view(b, c, k, k, h, w)
        out = torch.einsum('bcklhw,bklhw->bchw', [features_unfold, kernels])
        return out


class AdaptiveFeaturePropagation(nn.Module):
    def __init__(self, in_channels=1024):
        super(AdaptiveFeaturePropagation, self).__init__()
        self.kernel_weight_predictor = KernelWeightPredictor(in_channels)
        self.spatially_variant_convolution = SpatiallyVariantConvolution(kernel_size=9)

    def forward(self, current_frame_low_features, key_frame_low_features, key_frame_high_features):
        spatially_variant_kernels = self.kernel_weight_predictor(current_frame_low_features, key_frame_low_features)
        return self.spatially_variant_convolution(spatially_variant_kernels, key_frame_high_features)


class LowLatencyModel(nn.Module):
    def __init__(self, deeplab, threshold = 0.3, fixed_schedule = 5):
        super(LowLatencyModel, self).__init__()
        self.adaptive_feature_propagation = AdaptiveFeaturePropagation()
        self.adaptive_key_frame_selector = AdaptiveKeyFrameSelector()
        self.adapt_net = AdaptNet()
        self.deeplab = deeplab

        self.key_frame_low_features = None
        self.key_frame_high_features = None
        self.cur_frame_low_features = None
        self.steps_same_key_frame = 0
        self.fixed_schedule = fixed_schedule
        self.threshold = threshold

    def forward_deeplab(self, cur_frame_low_features):
        self.key_frame_low_features = cur_frame_low_features
        self.key_frame_high_features = self.deeplab.forward_high(cur_frame_low_features)
        return self.key_frame_high_features

    def compute_deviation(self, features_1, features_2):
        seg_map_1 = torch.argmax(features_1, dim=1)
        seg_map_2 = torch.argmax(features_2, dim=1)
        b, c, h, w = features_1.shape
        return torch.einsum('bhw->b', [torch.eq(seg_map_1, seg_map_2)]) / (h*w)


    def forward_train(self, input, random_input):
        random_frame_low_features = self.deeplab.forward_low(random_input)
        self.forward_deeplab(random_frame_low_features)
        cur_frame_low_features = self.deeplab.forward_low(input)

        # deviation
        deviation = self.adaptive_key_frame_selector(cur_frame_low_features, self.key_frame_low_features)
        cur_frame_high_features = self.deeplab.forward_high(cur_frame_low_features)
        real_deviation = self.compute_deviation(cur_frame_high_features, self.key_frame_high_features)

        # feature propagation
        x = self.adaptive_feature_propagation(cur_frame_low_features, self.key_frame_low_features,
                                              self.key_frame_high_features)
        x = self.adapt_net(x, cur_frame_low_features)

        return x, deviation, real_deviation


    def forward(self, input):
        cur_frame_low_features = self.deeplab.forward_low(input)
        if self.key_frame_low_features is None:
            return self.forward_deeplab(cur_frame_low_features)

        new_key_frame = False
        if self.fixed_schedule is not None:
            if self.steps_same_key_frame >= self.fixed_schedule:
                new_key_frame = True
        else:
            deviation = self.adaptive_key_frame_selector(cur_frame_low_features, self.key_frame_low_features)
            if deviation > self.threshold:
                new_key_frame = True

        if new_key_frame:
            return self.forward_deeplab(cur_frame_low_features)
        else:
            x = self.adaptive_feature_propagation(cur_frame_low_features, self.key_frame_low_features, self.key_frame_high_features)
            x = self.adapt_net(x, cur_frame_low_features)
            return x





