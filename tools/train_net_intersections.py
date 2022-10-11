#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
# Additionally modified by Yunqiu Xu for H2FA R-CNN
# ------------------------------------------------------------------------
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
import json
import gzip
from collections import OrderedDict
import torch
import time
import argparse
import contextlib

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data.build import build_detection_train_source_loader

try:
    _nullcontext = contextlib.nullcontext  # python 3.7+
except AttributeError:
    @contextlib.contextmanager
    def _nullcontext(enter_result=None):
        yield enter_result


cocodir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'MSCOCO2017'))
intersections_basedir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'Intersections'))
thing_classes_coco = [['person'], ['car', 'bus', 'truck']]
thing_classes = ['person', 'vehicle']
assert len(thing_classes_coco) == len(thing_classes)


def get_coco_dicts(split):
    if split == 'valid':
        annotations_json = os.path.join(cocodir, 'annotations', 'instances_val2017.json')
    elif split == 'train':
        annotations_json = os.path.join(cocodir, 'annotations', 'instances_train2017.json')
    else: return None
    with open(annotations_json, 'r') as fp:
        annotations = json.load(fp)
    category_id_remap = {}
    for cat in annotations['categories']:
        for i in range(0, len(thing_classes_coco)):
            if cat['name'] in thing_classes_coco[i]:
                category_id_remap[cat['id']] = i
    coco_dicts = {}
    images_dir = os.path.join(cocodir, 'images', 'val2017' if split == 'valid' else 'train2017')
    for im in annotations['images']:
        coco_dicts[im['id']] = {'file_name': os.path.join(images_dir, im['file_name']), 'image_id': im['id'], 'height': im['height'], 'width': im['width'], 'annotations': []}
    for ann in annotations['annotations']:
        if not ann['category_id'] in category_id_remap:
            continue
        coco_dicts[ann['image_id']]['annotations'].append({'bbox': ann['bbox'], 'iscrowd': ann['iscrowd'], 'bbox_mode': BoxMode.XYWH_ABS, 'segmentation': [], 'category_id': category_id_remap[ann['category_id']]})
    coco_dicts = list(coco_dicts.values())
    coco_dicts = list(filter(lambda x: len(x['annotations']) > 0, coco_dicts))
    for i in range(0, len(coco_dicts)):
        coco_dicts[i]['image_id'] = i + 1
    count_images, count_bboxes = len(coco_dicts), sum(map(lambda ann: len(ann['annotations']), coco_dicts))
    print('MSCOCO-2017 %s: %d images, %d bboxes' % (split, count_images, count_bboxes))
    return coco_dicts


def get_pseudo_dicts(args):
    lmdb_path = os.path.normpath(os.path.join(intersections_basedir, 'images', 'train_lmdb', args.id))
    with open(os.path.join(lmdb_path, 'frames.json'), 'r') as fp:
        meta = json.load(fp)
    ifilelist = meta['ifilelist']
    dict_json = []
    for i in range(0, len(ifilelist)):
        dict_json.append({'file_name': os.path.normpath(os.path.join(lmdb_path, 'jpegs', ifilelist[i])), 'image_id': i, 'height': meta['meta']['video']['H'], 'width': meta['meta']['video']['W'], 'annotations': []})
    with gzip.open(os.path.join(lmdb_path, 'detect_r101-fpn-3x.json.gz'), 'rt') as fp:
        dets = json.loads(fp.read())['dets']
    assert len(dets) == len(dict_json), 'detection & dataset mismatch'
    for i in range(0, len(dets)):
        for j in range(0, len(dets[i]['score'])):
            if dets[i]['score'][j] < 0.5:
                continue
            dict_json[i]['annotations'].append({'bbox': dets[i]['bbox'][j], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': dets[i]['label'][j], 'src': 'det', 'score': dets[i]['score'][j]})
    print('training frames of video %s at %s: %d images' % (args.id, lmdb_path, len(dict_json)))
    return dict_json


def get_manual_dicts(args):
    inputdir = os.path.join(intersections_basedir, 'images', 'annotated', args.id)
    with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
        annotations = json.load(fp)
    for i in range(0, len(annotations)):
        annotations[i]['file_name'] = os.path.join(inputdir, 'masked', annotations[i]['file_name'])
        annotations[i]['image_id'] = i + 1
    print('manual annotation for %s: %d images, %d bboxes' % (args.id, len(annotations), sum(list(map(lambda x: len(x['annotations']), annotations)))))
    return annotations


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.source_data_loader = self.build_train_source_loader(cfg)
        self._source_data_loader_iter = iter(self.source_data_loader)

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        while True:
            data = next(self._data_loader_iter)

            if all([len(x["instances"]) > 0 for x in data]):
                break

        source_data = next(self._source_data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data+source_data)
        losses = sum(loss_dict.values())
    
        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()

        # use a new stream so the ops don't wait for DDP
        with torch.cuda.stream(
            torch.cuda.Stream()
        ) if losses.device.type == "cuda" else _nullcontext():
            metrics_dict = loss_dict
            metrics_dict["data_time"] = data_time
            self._write_metrics(metrics_dict)
            self._detect_anomaly(losses, loss_dict)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()

    @classmethod
    def build_train_source_loader(cls, cfg):
        return build_detection_train_source_loader(cfg)
        
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        assert evaluator_type == "coco"
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def main(args):
    config_yaml = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'configs', 'CrossDomain-Detection', 'h2fa_rcnn_R_101_FPN_intersections.yaml'))
    ckpt_path = os.path.join(intersections_basedir, 'models', 'mscoco2017_remap_r101-fpn-3x.pth')
    assert os.access(config_yaml, os.R_OK)
    assert os.access(ckpt_path, os.R_OK)
    print('load configuration from:', config_yaml)
    
    DatasetCatalog.register('mscoco2017_train_remap', lambda: get_coco_dicts('train'))
    DatasetCatalog.register('mscoco2017_valid_remap', lambda: get_coco_dicts('valid'))
    MetadataCatalog.get('mscoco2017_train_remap').thing_classes = thing_classes
    MetadataCatalog.get('mscoco2017_valid_remap').thing_classes = thing_classes
    MetadataCatalog.get('mscoco2017_valid_remap').evaluator_type = 'coco'

    dst_pseudo, dst_manual = 'intersection_pseudo_%s' % args.id, 'intersection_manual_%s' % args.id
    DatasetCatalog.register(dst_pseudo, lambda: get_pseudo_dicts(args))
    DatasetCatalog.register(dst_manual, lambda: get_manual_dicts(args))
    MetadataCatalog.get(dst_pseudo).thing_classes = thing_classes
    MetadataCatalog.get(dst_manual).thing_classes = thing_classes
    MetadataCatalog.get(dst_manual).evaluator_type = 'coco'

    cfg = get_cfg()
    cfg.merge_from_file(config_yaml)
    cfg.MODEL.WEIGHTS = ckpt_path
    cfg.DATASETS.TRAIN_SOURCE = ('mscoco2017_train_remap',)
    cfg.DATASETS.TRAIN = (dst_pseudo,)
    cfg.DATASETS.TEST = (dst_manual,) # debug
    #cfg.DATASETS.TEST = ('mscoco2017_valid_remap', dst_manual)

    iters, eval_interval, cfg.SOLVER.IMS_PER_BATCH = 100, 51, 1 # debug
    #iters, eval_interval = 20000, 4010
    lr, num_workers = 1e-4, 4

    cfg.DATALOADER.NUM_WORKERS = num_workers
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.WARMUP_ITERS = iters // 10
    cfg.SOLVER.GAMMA = 0.5
    cfg.SOLVER.STEPS = (iters // 3, iters * 2 // 3)
    cfg.SOLVER.MAX_ITER = iters
    cfg.TEST.EVAL_PERIOD = eval_interval
    cfg.freeze()
    print('load weights from:', cfg.MODEL.WEIGHTS)

    if not os.access(cfg.OUTPUT_DIR, os.W_OK):
        os.mkdir(cfg.OUTPUT_DIR)
    assert os.path.isdir(cfg.OUTPUT_DIR)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    m = trainer.model
    if isinstance(m, torch.nn.DataParallel) or isinstance(m, torch.nn.parallel.DistributedDataParallel):
        print('unwrap data parallel')
        m = m.module
    torch.save(m.state_dict(), os.path.join(os.path.dirname(__file__), 'adapt_intersections_%s_lr%.5f_iter%d.pth' % (args.id, cfg.SOLVER.BASE_LR, cfg.SOLVER.MAX_ITER)))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adaptation Script')
    parser.add_argument('--opt', type=str, help='option')
    parser.add_argument('--id', type=str, help='video ID')
    #parser.add_argument('--ddp_num_gpus', type=int, default=1)
    #parser.add_argument('--ddp_port', type=int, default=50152)
    args = parser.parse_args()

    main(args)
