import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from datasets.utils import decode_seg_map_sequence


class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, dataset, image, target, output, global_step):
        grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
        writer.add_image('Image', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
                                                       dataset=dataset), 3, normalize=False, range=(0, 255))
        writer.add_image('Predicted_label', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
                                                       dataset=dataset), 3, normalize=False, range=(0, 255))
        writer.add_image('Groundtruth_label', grid_image, global_step)

    def visualize_image_mulitview(self, writer, dataset, image, target, output_single, output_multiview, global_step):
        grid_image = make_grid(image[:4].clone().cpu().data, 4, normalize=True)
        writer.add_image('Image', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.max(output_single[:4], 1)[1].detach().cpu().numpy(),
                                                       dataset=dataset), 4, normalize=False, range=(0, 255))
        writer.add_image('Predicted_label_singleview', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.max(output_multiview[:4], 1)[1].detach().cpu().numpy(),
                                                       dataset=dataset), 4, normalize=False, range=(0, 255))
        writer.add_image('Predicted_label_multiview', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:4], 1).detach().cpu().numpy(),
                                                       dataset=dataset), 4, normalize=False, range=(0, 255))
        writer.add_image('Groundtruth_label', grid_image, global_step)
