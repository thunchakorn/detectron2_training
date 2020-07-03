import os
import shutil
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
from detectron2.engine import launch
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import launch



def main(args):
    train_name, test_name = regist_dataset(args.train_label_path, args.test_label_path)
    cfg, hyperparameters = setup(args, train_name, test_name)
    dest_dir = os.path.join(cfg.OUTPUT_DIR, 'sample_compare_result')
    if not args.resume:
        if os.path.isdir(cfg.OUTPUT_DIR):
            shutil.rmtree(cfg.OUTPUT_DIR)
        os.mkdir(cfg.OUTPUT_DIR)
        os.mkdir(dest_dir)

    mlflow.log_params(hyperparameters)
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_evaluate(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    model = do_train(cfg, model, resume=args.resume)
    mlflow.pytorch.log_model(pytorch_model = model,
                         artifact_path = 'model_best',
                         conda_env = mlflow.pytorch.get_default_conda_env())

    results = do_evaluate(cfg, model)
    mlflow.log_metrics({k + 'bbox':v for k,v in results['bbox'].items()})
    mlflow.log_metrics({k + 'segm':v for k,v in results['segm'].items()}) 
    
    dataset_val = load_coco_json(json_file = args.test_label_path,
                               image_root = '',
                               dataset_name = "financial_val")

    compare_gt(json_file = args.test_label_path,
    dataset_name = test_name,
    dest_dir = dest_dir,
    cfg = cfg,
    weight = os.path.join(cfg.OUTPUT_DIR, 'model_best.pth'),
    score_thres_test = 0.7,
    num_sample = 10
    )

    mlflow.log_artifacts(cfg.OUTPUT_DIR)


if __name__ == "__main__":

    args = default_argument_parser().parse_args()
    # mlflow.set_tracking_uri(remote_server_uri)
    # mlflow.set_experiment("/my-experiment")
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )