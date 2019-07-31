import cv2
import numpy as np
from PIL import Image
from matplotlib import gridspec
from matplotlib import pyplot as plt


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

    Returns:
      A colormap for visualizing segmentation results.
    """

    def bit_get(val, idx):
        """Gets the bit value.

        Args:
          val: Input value, int or numpy int array.
          idx: Which bit of the input val.

        Returns:
          The "idx"-th bit of input val.
        """
        return (val >> idx) & 1

    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3

    return colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
      label: A 2D array with integer type, storing the segmentation label.

    Returns:
      result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.

    Raises:
      ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


def vis_segmentation(image, seg_map, fig=None):
    """Visualizes input image, segmentation map and overlay view."""
    # plt.ion()
    if fig is None:
        fig = plt.figure(figsize=(15, 5))
    seg_image, overlay_image = label2rgb(seg_map, image)
    grid_spec = gridspec.GridSpec(1, 3, width_ratios=[6, 6, 6])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')

    plt.subplot(grid_spec[1])
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')

    plt.subplot(grid_spec[2])
    plt.imshow(overlay_image)
    plt.axis('off')
    plt.title('segmentation overlay')

    # unique_labels = np.unique(seg_map)
    # ax = plt.subplot(grid_spec[3])
    # plt.imshow(FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    # ax.yaxis.tick_right()
    # plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
    # plt.xticks([], [])
    # ax.tick_params(width=0.0)
    # plt.grid('off')
    # fig.canvas.draw()
    # plt.close()
    # plt.ion()
    # plt.show()
    # plt.pause(0.01)
    plt.ion()
    # plt.imshow(data)
    plt.show()
    plt.pause(0.01)
    return fig


def fig2img(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)  # COLOR_BGR2RGB
    return data


def label2rgb(lbl, img=None, n_labels=None, alpha=0.5, colors=None):
    mask = create_pascal_label_colormap()[lbl].astype(np.uint8)

    if img is not None:
        print(img.dtype, img.shape)
        img_gray = Image.fromarray(img).convert('LA')
        img_gray = np.asarray(img_gray.convert('RGB'))
        overlay = alpha * mask + (1 - alpha) * img_gray
        overlay = overlay.astype(np.uint8)
    else:
        overlay = mask

    return mask, overlay
