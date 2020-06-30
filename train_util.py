

# import some common libraries
import numpy as np
import cv2
import random
import os
import time
import torch
import copy
from collections import OrderedDict
import time

# import mlflow
from mlflow import log_metric, log_param, log_artifact

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
from detectron2 import config
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetMapper, MapDataset
from detectron2.data import build_detection_test_loader
from detectron2.data.detection_utils import read_image
from detectron2.data.datasets import load_coco_json, register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)



def do_test(cfg, model):
    """
    Evaluate on test set using coco evaluate
    """
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = COCOEvaluator(cfg.DATASETS.TEST, cfg, False, output_dir= cfg.OUTPUT_DIR)
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results

def val_mapper(dataset_dict):
    """
    mapper function for do_val function
    """
    mapper = DatasetMapper(cfg, True) #True for return ground truth
    return mapper(dataset_dict)

def do_val(cfg, model, data_val_loader, storage):
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

    data_loader = build_detection_train_loader(cfg)
    
    best_val_loss = None
    data_val_loader = build_detection_test_loader(cfg,
                                                  cfg.DATASETS.TEST[0],
                                                  mapper = val_mapper)
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
                val_total_loss = do_val(cfg, model, data_val_loader, storage)
                logger.setLevel(logging.DEBUG)
                logger.info(f"validation loss of iteration {iteration}th: {val_total_loss}")
                storage.put_scalar(name = 'val_total_loss', value = val_total_loss)
                
                if best_val_loss is None or val_total_loss < best_val_loss:
                  best_val_loss = val_total_loss
                  best_model_weight = copy.deepcopy(model.state_dict())

                comm.synchronize()
            
            # สร้าง checkpointer เพิ่มให้ save best model โดยดูจาก val loss
            if iteration - start_iter > 5 and (iteration % 20 == 0 or iteration == max_iter):
                logger.info(f"time per iteration: {(time.time() - start)/20} seconds")
                for writer in writers:
                    writer.write()
            
    model.load_state_dict(best_model_weight)
    checkpointer.save('model_best')     
    do_test(cfg, model)
            # periodic_checkpointer.step(iteration - 1) 

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg


