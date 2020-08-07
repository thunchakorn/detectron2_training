import argparse
# import mlflow
import mlflow 
import mlflow.pytorch

from train_util import do_evaluate, compare_gt, setup
from detectron2.config import get_cfg
from detectron2.modeling import build_model


def main(args):
    test_name, num_class = regist_dataset(args.test_label_path, args.thing_classes)
    cfg, hyperparameters = setup(args, None, test_name, num_class)
    dest_dir = os.path.join(cfg.OUTPUT_DIR, 'sample_compare_result')

    if not os.path.isdir(cfg.OUTPUT_DIR):
        os.mkdir(cfg.OUTPUT_DIR)
        if not os.path.isdir(dest_dir):
            os.mkdir(dest_dir)
    
    mlflow.log_params(hyperparameters)
    model = build_model(cfg)
    results = do_evaluate(cfg, model)
    mlflow.log_metrics({k + '_bbox':v for k,v in results['bbox'].items()})
    mlflow.log_metrics({k + '_segm':v for k,v in results['segm'].items()}) 
    print(results)
    compare_gt(cfg, dir = args.test_label_path)
    mlflow.log_artifacts(dest_dir)


if __name__ == "__main__":
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    mlflow.set_experiment(os.getenv('MLFLOW_EXPERIMENT_NAME'))
    parser = argparse.ArgumentParser()

    with mlflow.start_run():
        args = parser.parse_args()
        
        main(args)



    