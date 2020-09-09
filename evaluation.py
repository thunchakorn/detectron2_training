import argparse
import os
# import mlflow
import mlflow 
import mlflow.pytorch

from train_util import do_evaluate, compare_gt, setup, regist_dataset, regist_coco_dataset
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

parser = argparse.ArgumentParser()

parser.add_argument('-w',
'--weight',
type=str,
required=True,
help='path to weight .pth file')

parser.add_argument('-a',
'--annotation',
type=str,
required=True,
help='path to images annotation file .json in coco format')

parser.add_argument('-th',
'--thing_classes',
type=str,
required=True,
help='path to class list file')

parser.add_argument('-c',
'--config_file',
type=str,
required=True,
help='path to config file in model zoo e.g. /COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml')

parser.add_argument('-s',
'--score_test_thres',
default = 0.7,
type = float,
help = 'cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = ...')

args = parser.parse_args()


def main(args):
    test_name, num_class = regist_coco_dataset(args.annotation, args.thing_classes)
    cfg, _ = setup(args, None, test_name, num_class)
    cfg.MODEL.WEIGHTS = args.weight
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_test_thres
    dest_dir = os.path.join(cfg.OUTPUT_DIR, 'sample_compare_result')

    if not os.path.isdir(cfg.OUTPUT_DIR):
        os.mkdir(cfg.OUTPUT_DIR)
    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)
    
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    results = do_evaluate(cfg, model)
    mlflow.log_metrics({k + '_bbox':v for k,v in results['bbox'].items()})
    mlflow.log_metrics({k + '_segm':v for k,v in results['segm'].items()}) 
    print(results)

    compare_gt_coco(cfg, annotation_file = args.annotation,
    dest_dir = dest_dir,
    weight = None,
    score_thres_test = args.score_test_thres,
    num_sample = num_class
    )
    mlflow.log_artifacts(dest_dir)


if __name__ == "__main__":
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    mlflow.set_experiment(os.getenv('MLFLOW_EXPERIMENT_NAME'))
    with mlflow.start_run():
        
        main(args)



    