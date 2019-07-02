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


# class KernelWeightPredictor(nn.Module):
#     '''
#     Produces the weights used for the Spatially Variant Convolution
#     '''
#     def __init__(self, in_channels=1024):
#         self.conv_reduce = nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=3, padding=1)
#         self.conv_2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
#         self.conv_3 = nn.Conv2d(in_channels=256, out_channels=81, kernel_size=1)
#         pass
#
#     def forward(self, current_frame_low_features, key_frame_low_features):
#         cur = F.relu(self.conv_reduce(current_frame_low_features))
#         key = F.relu(self.conv_reduce(key_frame_low_features))
#         out = F.relu(self.conv_2(torch.cat((cur, key), 1)))
#         out = F.softmax(out, 1)  # output dim = N x (9**2) x W x H
#         out = out.view(-1, 9,9, out.size()[2], out.size()[3])

# class AdaptiveFeaturePropagation(nn.Module):
#     def __init__(self, in_channels=1024):
#         self.kernel_weight_predictor = KernelWeightPredictor(in_channels)
#
#     def forward(self, current_frame_low_features, key_frame_low_features, key_frame_high_features):
#         spatially_variant_kernels = self.kernel_weight_predictor(current_frame_low_features, key_frame_low_features)
#
#         out = torch.zeros(key_frame_high_features.size())
#         temp = F.pad(key_frame_high_features, (0,0,4, 4), 'constant', 0)
#         for i in range(N):
#             for w_i in range(W):
#                 for h_i range(H):
#                     for c_i in range(C):
#                         out[i, c_i, w_i, h_i] =
#
#         return torch.cat([F.conv2d()], 0)


# rep_flow version
class AdaptiveFeaturePropagation(nn.Module):
    def __init__(self, in_channels=1024, n_iter=5, learnable=True, flow_channels=32):
        super(AdaptiveFeaturePropagation, self).__init__()
        self.conv_reduce = nn.Conv2d(in_channels=in_channels, out_channels=flow_channels, kernel_size=3, padding=1)
        self.flow_layer = FlowLayer(channels=flow_channels, n_iter=n_iter, params=learnable)
        self.conv_expand = nn.Conv2d(in_channels=flow_channels, out_channels=in_channels, kernel_size=3, padding=1)
        self.conv_mix_0 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        self.conv_mix_1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)

    def forward(self, current_frame_low_features, key_frame_low_features, key_frame_high_features):
        reduce_key = F.relu(self.conv_reduce(key_frame_low_features))
        reduce_cur = F.relu(self.conv_reduce(current_frame_low_features))
        x = torch.cat((reduce_key, reduce_cur), 2)
        b, c, t, h, w = x.size()
        u, v = self.flow_layer(x[:, :, :-1].permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w),
                               x[:, :, 1:].permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w))
        x = torch.cat([u, v], dim=1)
        x = x.view(b, t - 1, c * 2, h, w).permute(0, 2, 1, 3, 4).contiguous().squeeze() #shape BxC*2xHxW
        x = F.relu(self.conv_expand(x))
        x = torch.cat((x, key_frame_high_features), 1)
        x = F.relu(self.conv_mix_0(x))
        x = F.relu(self.conv_mix_1(x))
        return x

class LowLatencyModel(nn.Module):
    def __init__(self, s_l, s_h, output_net, threshold = 0.3):
        super(LowLatencyModel, self).__init__()
        self.adaptive_feature_propagation = AdaptiveFeaturePropagation()
        self.adaptive_key_frame_selector = AdaptiveKeyFrameSelector()
        self.s_l = s_l
        self.s_h = s_h
        self.output_net = output_net

        self.key_frame_low_features = None
        self.key_frame_high_features = None

    def forward_all_the_way(self, cur_frame_low_features):
        self.key_frame_low_features = cur_frame_low_features
        self.key_frame_high_features = self.s_h(self.key_frame_low_features)
        return self.output_net(self.key_frame_high_features)

    def forward(self, input):
        cur_frame_low_features = self.s_l(input)
        if self.key_frame_low_features == None:
            return self.forward_all_the_way(cur_frame_low_features)

        deviation = self.adaptive_key_frame_selector()
        if deviation > threshold:




