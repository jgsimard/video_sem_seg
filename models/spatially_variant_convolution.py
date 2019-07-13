import torch

def spatially_variant_convolution(features, kernels):
    '''
    Explained in this paper : https://arxiv.org/abs/1804.00389 at equation (1)
    :param features: shape=BxCxHxW
    :param kernels: BxK^2xHxW
    :return: shape=BxCxHxW
    '''
    temp_kernels = torch.unsqueeze(kernels, 1) # now shape=Bx1xK^2xHxW