import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import torch.autograd as autograd
from torch.autograd.variable import Variable

HEIGHT = 287
WIDTH = 352


class Discriminator(nn.Module):
    def __init__(self, input_nc, img_height, img_width, filter_base=16, num_block=6, n_iter=500,
                 generator_loss_weight=2, gp_weigth=10, lr_ratio=0.1):
        super(Discriminator, self).__init__()

        self.n_iter = n_iter
        self.generator_loss_weight = generator_loss_weight
        self.lr_ratio = lr_ratio
        self.gp_weigth = gp_weigth

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        model = []
        model += discriminator_block(input_nc, filter_base, bn=False)

        for i in range(num_block - 1):
            model += discriminator_block(filter_base * 2 ** i, filter_base * 2 ** (i + 1), bn=True)

        self.model = nn.Sequential(*model)

        height = img_height // 2 ** num_block + 1
        width = img_width // 2 ** num_block + 1
        self.adv_layer = nn.Linear(filter_base * 2 ** (i + 1) * height * width, 1)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


def compute_gradient_penalty(D, x_real, x_fake):
    a = Tensor(x_real.shape[0], 1, 1, 1).uniform_(0, 1).expand_as(x_real).cuda().detach()
    z = (a * x_real + (1 - a) * x_fake).requires_grad_(True)
    D_z = D(z)
    grads = autograd.grad(outputs=D_z,
                          inputs=z,
                          grad_outputs=torch.ones_like(D_z).cuda(),
                          create_graph=True,
                          retain_graph=True,
                          only_inputs=True, )[0]
    gradient_penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def label_smoothing(target, n_epoch, total_epoch):
    """

    :param target:
        [B, C, H, W]
    :return:
        [B, C, H, W]
    """
    # set 0 to random value in range (0.0, 0.3)
    # set 1 to random value in range (0.75, 1.25)

    batch, channel, height, width = target.shape

    target_smoothed = target.clone()
    target_smoothed[target == 1] = (torch.rand(target[target == 1].shape).cuda() - 0.5) * 0.3 * pow(
        (1 - n_epoch / total_epoch),
        0.9) + 1.0
    target_smoothed[target == 0] = torch.rand(target[target == 0].shape).cuda() * 0.3 * pow((1 - n_epoch / total_epoch),
                                                                                            0.9) + 0.0

    target_smoothed /= torch.sum(target_smoothed, dim=1).view(batch, 1, height, width)

    return target_smoothed


def onehot(targets, num_classes):
    """
    :param targets: index tensor
    :param num_classes: number of classes
    """
    # assert isinstance(targets, torch.LongTensor)
    batch, height, width = targets.shape
    target_onehot = torch.zeros(batch, num_classes, height, width).cuda()
    return target_onehot.scatter_(1, targets.view(batch, 1, height, width).long(), 1)


if __name__ == '__main__':
    import torchsummary

    # target = torch.ones([1, 5, 5]).cuda()
    # target_onehot = onehot(target, 5)
    # target_smoothed = label_smoothing(target_onehot, 50)

    d = Discriminator(13, HEIGHT, WIDTH, filter_base=16).cuda()

    torchsummary.summary(d, (13, 287, 352))

    print(d)

    batch = 5
    num_class = 13

    fake_input = torch.rand([batch, num_class, HEIGHT, WIDTH]).cuda()
    fake_input = torch.softmax(fake_input, dim=1).requires_grad_(False)
    fake_validity = d(fake_input).requires_grad_(True)

    real_input = torch.randint(0, num_class - 1, [batch, HEIGHT, WIDTH]).cuda()
    real_input = onehot(real_input, num_class)
    real_input = label_smoothing(real_input, 10, 50).requires_grad_(False)
    real_validity = d(real_input).requires_grad_(True)

    loss = - real_validity.mean() + fake_validity.mean() + 10 * wgan_gp_loss(d, real_input, fake_input)

    loss.backward()

    print(loss)
