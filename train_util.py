# import some common libraries
import numpy as np
import cv2
import random
import os
import sys
import time
import torch
import copy
from collections import OrderedDict
import time
import argparse
import logging

# import mlflow
import mlflow 
import mlflow.pytorch

# import some common detectron2 utilities
import detectron2.utils.comm as comm
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
    )
from detectron2.engine import default_setup, DefaultPredictor
from detectron2 import config
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetMapper, MapDataset
from detectron2.data import transforms as T 
from detectron2.data.detection_utils import read_image
from detectron2.data.datasets import load_coco_json, register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
from detectron2.utils.visualizer import Visualizer

from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)

logger = logging.getLogger("detectron2")

def do_evaluate(cfg, model):
    """
    Evaluate on test set using coco evaluate
    """
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = COCOEvaluator(dataset_name, cfg, False, output_dir= cfg.OUTPUT_DIR)
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def do_val_monitor(cfg, model, data_val_loader):
    """
    get loss of validate/test set for monitoring in do_train function
    """
    total_val_loss = []
    with torch.no_grad():
        for data in data_val_loader: #load batch of all validate set
            loss_dict = model(data)
            losses = sum(loss_dict.values())
            total_val_loss.append(losses)
    return sum(total_val_loss)/len(total_val_loss) #averaging loss

def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )
    min_size = cfg.INPUT.MIN_SIZE_TRAIN 
    max_size = cfg.INPUT.MAX_SIZE_TRAIN, 
    sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    data_loader = build_detection_train_loader(cfg, mapper=DatasetMapper(cfg,
                                                                        is_train=True,
                                                                        augmentations=[
                                                                        T.ResizeShortestEdge(min_size, max_size, sample_style),
                                                                        T.RandomFlip(prob = 0.5, vertical = False),
                                                                        T.RandomRotation(angle = [-10.0, 10.0]),
                                                                        
   ]))
    best_model_weight = copy.deepcopy(model.state_dict())
    best_val_loss = None
    data_val_loader = build_detection_test_loader(cfg,
                                                  cfg.DATASETS.TEST[0],
                                                  mapper = DatasetMapper(cfg, True))
    logger.info("Starting training from iteration {}".format(start_iter))

    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            iteration += 1
            start = time.time()
            storage.step()

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and iteration % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter
            ):
                logger.setLevel(logging.CRITICAL)
                print('validating')
                val_total_loss = do_val_monitor(cfg, model, data_val_loader)
                logger.setLevel(logging.DEBUG)
                logger.info(f"validation loss of iteration {iteration}th: {val_total_loss}")
                storage.put_scalar(name = 'val_total_loss', value = val_total_loss)
                
                if best_val_loss is None or val_total_loss < best_val_loss:
                  best_val_loss = val_total_loss
                  best_model_weight = copy.deepcopy(model.state_dict())

                comm.synchronize()
            
            # สร้าง checkpointer เพิ่มให้ save best model โดยดูจาก val loss
            if iteration - start_iter > 5 and (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            
    model.load_state_dict(best_model_weight)
    checkpointer.save('model_best')
    return model
                         
def regist_dataset(json_train_path, json_test_path):
    """
    register training and testing dataset
    dataset must be in coco format
    ouput: name of training and testing set
    """
    train_name = os.path.split(json_train_path)[-1][:-5]
    test_name = os.path.split(json_test_path)[-1][:-5]
    
    register_coco_instances(train_name,
                            {},
                            json_train_path,
                            "")
    
    register_coco_instances(test_name,
                            {},
                            json_test_path,
                            "")
    return train_name, test_name

def compare_gt(json_file, cfg, dataset_name, weight, dest_dir, score_thres_test = 0.7, num_sample = 10):
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thres_test
  cfg.MODEL.WEIGHTS = weight
  predictor = DefaultPredictor(cfg)

  dataset_list_dict = load_coco_json(json_file,
                                  image_root = '',
                                  dataset_name = dataset_name)
  
  if len(dataset_list_dict) > num_sample:
    sample = random.sample(range(len(dataset_list_dict)), num_sample)
  else:
    sample = range(len(dataset_list_dict))
  for s in sample:
    img_dict = dataset_list_dict[s]
    print(img_dict['file_name'])
    img = read_image(img_dict['file_name'],format = 'BGR')
    h, w = img_dict['height'], img_dict['width']
    v_gt = Visualizer(img[:, :, ::-1],
                            metadata=MetadataCatalog.get(dataset_name),
                            scale=0.5)
    v_gt = v_gt.draw_dataset_dict(img_dict)

    #predicting
    outputs = predictor(img)

    #visualizing frmo prediction result
    
    v_pd = Visualizer(img[:, :, ::-1], MetadataCatalog.get(dataset_name), scale=1.2)
    v_pd = v_pd.draw_instance_predictions(outputs["instances"].to("cpu"))

    gt = cv2.resize(v_gt.get_image()[:, :, ::-1], (w,h))
    pd = cv2.resize(v_pd.get_image()[:, :, ::-1], (w,h))

    #stacking groudtruth and prediction
    merge_img = np.hstack((gt, pd))
    result_name = os.path.join(dest_dir, os.path.split(img_dict['file_name'])[1])
    cv2.imwrite(result_name, merge_img)

def default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
    Examples:

    Run on single machine:
        $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml -trp ./train.json -tep ./test.json MODEL.WEIGHTS /path/to/weight.pth

    Run on multiple machines:
        (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
        (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
    """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )
    parser.add_argument("--mlflow_path", help="url to log mlflow", default = None)

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )

    parser.add_argument(
        '-trp',
        '--train_label_path',
        required = True,
        help = 'path to train label json file in coco format e.g. ./train.json',
        default = './train.json',
        type = str
    )

    parser.add_argument(
        '-tep',
        '--test_label_path',
        required = True,
        help = 'path to test label json file in coco format e.g. ./test.json',
        default = './test.json',
        type = str
    )

    parser.add_argument(
        "opts",
        help="""Other configs from https://detectron2.readthedocs.io/modules/config.html#config-references
        non string and numeric type must in quotation mark "" e.g. SOLVER.STEPS '(2000, )'
        """,
        default = [],
        nargs = argparse.REMAINDER,
    )
    return parser

def setup(args, train_name, test_name):
    """
    Create configs and perform basic setups and log hyperparameter
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.DATASETS.TRAIN = (train_name, )
    cfg.DATASETS.TEST = (test_name, )
    # cfg.freeze()
    default_setup(
        cfg, args
    )
    # get adjusted hyperparameter  
    hyperparameters = {i:k for i,k in zip(args.opts[0::2], args.opts[1::2])}
    return cfg, hyperparameters

























