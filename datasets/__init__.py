from datasets import cityscapes, coco, combine_dbs, pascal, sbd, isi
from datasets import isi
from datasets import isi_multiview
from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):

    if args.dataset == 'pascal':
        train_set = pascal.VOCSegmentation(args, split='train')
        val_set = pascal.VOCSegmentation(args, split='val')
        if args.use_sbd:
            sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
            train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'cityscapes':
        train_set = cityscapes.CityscapesSegmentation(args, split='train')
        val_set = cityscapes.CityscapesSegmentation(args, split='val')
        test_set = cityscapes.CityscapesSegmentation(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'coco':
        train_set = coco.COCOSegmentation(args, split='train')
        val_set = coco.COCOSegmentation(args, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == "camvid":
        raise NotImplementedError

    elif args.dataset == "isi_rgb":
        train_set = isi.DeepSightRGB(root_dir=args.dataset_dir, split="train", hd=args.hd)
        val_set = isi.DeepSightRGB(root_dir=args.dataset_dir, split="validation", hd=args.hd)
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == "isi_rgb_temporal":
        train_set = isi.DeepSightTemporalRGB(root_dir=args.dataset_dir, split="train", train_range=args.train_distance)
        val_set = isi.DeepSightTemporalRGB(root_dir=args.dataset_dir, split="validation", train_range=args.train_distance)
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == "isi_intensity":
        train_set = isi.DeepSightDepth(root_dir=args.dataset_dir, split="train")
        val_set = isi.DeepSightDepth(root_dir=args.dataset_dir, split="validation")
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == "isi_multiview":
        train_set = isi_multiview.DeepSightDepthMultiview(root_dir=args.dataset_dir, split="train")
        val_set = isi_multiview.DeepSightDepthMultiview(root_dir=args.dataset_dir, split="validation")
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class
    elif args.dataset == "isi_multiview_2018":
        train_set = isi_multiview.DeepSightDepthMultiview(root_dir=args.dataset_dir, split="train")
        val_set = isi_multiview.DeepSightDepthMultiview(root_dir=args.dataset_dir, split="validation")
        num_class = 10
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError

