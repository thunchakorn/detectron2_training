import argparse
import pickle
import json
import glob
import numpy as np
import os
import cv2

import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_test_loader, DatasetCatalog
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultPredictor


def find_polygon(mask, percent = 0.005):
    mask = np.uint8(mask)
    contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    eps = percent * cv2.arcLength(contour[0], True)
    out_curve = cv2.approxPolyDP(contour[0], eps, False)
    return out_curve.reshape(-1,2).tolist()

def gen_predict(cfg, jpg_files, thing_classes):
    predictor = DefaultPredictor(cfg)
    
    for d in jpg_files:
        imagePath = os.path.split(d)[-1]
        imageData = None
        shapes = []

        img = cv2.imread(d)
        output = predictor(img)['instances'].to("cpu")
        inst = output.get_fields()
        imageHeight = output.image_size[0]
        imageWidth = output.image_size[1]

        for p, m in zip(inst['pred_classes'].numpy(), inst['pred_masks'].numpy()):
            polygon = find_polygon(m)
            dict_predict = {'shape_type':'polygon',
                            'group_id':None,
                            'label':thing_classes[p],
                            'points':polygon,
                            'flags':{}
                            }
            shapes.append(dict_predict)

        json_file = {'imagePath':imagePath,
                    'imageHeight':imageHeight,
                    'imageWidth':imageWidth,
                    'imageData':imageData,
                    'shapes':shapes
                    }

        file_name = d[:-3] + 'json'
        with open(file_name, 'w') as j:
            json.dump(json_file, j)

def main(weight, directory, thing_classes, config_file, thres):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.MODEL.WEIGHTS = args.weight
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thres
    jpg_files = glob.glob(os.path.join(directory, "*.jpg"))

    with open('classes.txt', 'r') as f:
        thing_classes = []
        for line in f.readlines():
            thing_classes.append(line.strip())
    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
    gen_predict(cfg, jpg_files, thing_classes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-w',
    '--weight',
    type=str,
    required=True,
    help='path to weight .pth file')

    parser.add_argument('-d',
    '--directory',
    type=str,
    required=True,
    help='image directory file')

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
    help = 'cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = ...'
    )

    args = parser.parse_args()

    main(args.weight, args.directory, args.thing_classes, args.config_file, args.score_test_thres)
