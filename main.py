import os
import shutil
import logging

# import mlflow
import mlflow 
import mlflow.pytorch

from train_utils import *
from torch.nn.parallel import DistributedDataParallel

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.utils.comm as comm
from detectron2.engine import launch
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import launch


def main(args):
    train_name, num_class = regist_coco_dataset(args.train_annotation, args.thing_classes)
    test_name, _ = regist_coco_dataset(args.test_annotation, args.thing_classes)
    cfg, hyperparameters = setup(args, train_name, test_name, num_class)
    dest_dir = os.path.join(cfg.OUTPUT_DIR, 'sample_compare_result')
    if not args.resume:
        if os.path.isdir(cfg.OUTPUT_DIR):
            shutil.rmtree(cfg.OUTPUT_DIR)
        os.mkdir(cfg.OUTPUT_DIR)
        os.mkdir(dest_dir)
    if hasattr(args, 'opts'):
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
    mlflow.log_metrics({k + '_bbox':v for k,v in results['bbox'].items()})
    mlflow.log_metrics({k + '_segm':v for k,v in results['segm'].items()}) 
    experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME')
    
    compare_gt_coco(cfg, annotation_file = args.test_annotation,
    dest_dir = dest_dir,
    weight = os.path.join(cfg.OUTPUT_DIR, f'model_{experiment_name}.pth'),
    score_thres_test = 0.7,
    num_sample = num_class
    )

    mlflow.log_artifacts(dest_dir)


if __name__ == "__main__":
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    mlflow.set_experiment(os.getenv('MLFLOW_EXPERIMENT_NAME'))
    with mlflow.start_run():
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