!detectron2_training

!!run docker
<code>docker run --gpus all -it --rm \
-v $(pwd)/data:/app/data \
-p 0.0.0.0:7007:6006 detectron2_training</code>

!!run evaluation
<code>
python evaluation.py \
-w data/weight.pth \
-c COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml \
-a data/results/test.json \
-t data/classes.txt \
-s 0.7 
</code>

!!run training
<code>
python main.py \
--config_file COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml \
-tra 'data/results/train_split.json' \
-va 'data/results/val_split.json' \
-tea 'data/results/test.json' \
-th data/classes.txt \
MODEL.WEIGHTS data/weight.pth \
TEST.EVAL_PERIOD 200 \
DATALOADER.NUM_WORKERS 1 \
SOLVER.IMS_PER_BATCH  1 \
SOLVER.BASE_LR 0.001 \
SOLVER.MAX_ITER 2000 \
SOLVER.WARMUP_ITERS 1000 \
SOLVER.STEPS '(4000,)' \
SOLVER.GAMMA 0.1 
</code>