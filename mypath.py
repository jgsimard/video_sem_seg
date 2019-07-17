class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/path/to/datasets/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        elif dataset =="isi":
            return '/home/deepsight/data/rgb'
        elif dataset =="isi_intensity":
            return '/home/deepsight/data/sem_seg_multiview_07_10_2019'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
