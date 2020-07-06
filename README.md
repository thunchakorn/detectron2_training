detectron2_training

<code>!python ./detectron2_training/main.py \
--config-file ./detectron2_training/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml \
-trp ./train.json \
-tep ./val.json \
MODEL.WEIGHTS ./model_final_trimmed.pth \
MODEL.ROI_HEADS.NUM_CLASSES  2 \
TEST.EVAL_PERIOD  50 \
DATALOADER.NUM_WORKERS  2 \
SOLVER.IMS_PER_BATCH  2 \
SOLVER.BASE_LR 0.001 \
SOLVER.MAX_ITER 100 \
SOLVER.WARMUP_ITERS 20 \
SOLVER.STEPS '(90,)' \
SOLVER.GAMMA 0.1 </code>
