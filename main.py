import os
import logging
#Requirements for detectron2

#Linux or macOS with Python ≥ 3.6
#PyTorch ≥ 1.4
#torchvision that matches the PyTorch installation. You can install them together at pytorch.org to make sure of this.
#pycocotools. Install it by pip install pycocotools>=2.0.1.
#OpenCV, optional, needed by demo and visualization

from train_util import *
from torch.nn.parallel import DistributedDataParallel

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.utils.comm as comm
from detectron2.engine import default_argument_parser, launch
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import launch


def main(args):
    train_name, test_name = regist_dataset(args.train_json, args.test_json)
    cfg = setup(args, train_name, test_name)
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg, model, resume=args.resume)
    return do_test(cfg, model)


if __name__ == "__main__":
    logger = logging.getLogger("detectron2")

    args = default_argument_parser().parse_args()
    
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
    compare_gt