import os
import cv2
import random
import json
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

register_coco_instances("car-project", {}, "/home/ms/sotsuken/carfusion_to_coco/carfusion_datasets/carfusion/annotations/car_keypoints_test.json", "/home/ms/sotsuken/carfusion_to_coco/carfusion_datasets/carfusion/test")
keypoint_names = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13']
keypoint_flip_map = []
keypoint_connection_rules = [('0','1',(255,0,0)),('4','5',(255,0,0)),('9','10',(255,0,0)),
                             ('2','3',(0,0,255)),('6','7',(0,0,255)),('11','12',(0,0,255)),
                             ('1','3',(255,0,255)),('3','7',(255,0,255)),('7','12',(255,0,255)),('12','10',(255,0,255)),('10','5',(255,0,255)),('5','1',(255,0,255)),
                             ('0','2',(255,255,0)),('2','6',(255,255,0)),('6','11',(255,255,0)),('11','9',(255,255,0)),('9','4',(255,255,0)),('4','0',(255,255,0))]      


'''


register_coco_instances("car-project", {}, "/home/ms/Downloads/cartest.json", "/home/ms/sotsuken/coco-annotator/datasets/cartest")
keypoint_names = ["FU","FD","RU","RD"]
keypoint_flip_map = []
keypoint_connection_rules = []
'''
MetadataCatalog.get("car-project").thing_classes = ["car"]
MetadataCatalog.get("car-project").thing_dataset_id_to_contiguous_id = {1:0}
MetadataCatalog.get("car-project").keypoint_names = keypoint_names
MetadataCatalog.get("car-project").keypoint_flip_map = keypoint_flip_map
MetadataCatalog.get("car-project").keypoint_connection_rules = keypoint_connection_rules

car_metadata = MetadataCatalog.get("car-project")
dataset_dicts = DatasetCatalog.get("car-project")
metadata = MetadataCatalog.get("car-project")
dataset_dicts = DatasetCatalog.get("car-project")


#教師データ確認
for d in random.sample(dataset_dicts, 1):
  img = cv2.imread(d["file_name"])
  visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
  vis = visualizer.draw_dataset_dict(d)
  cv2.imshow("sample",vis.get_image()[:, :, ::-1])
  cv2.waitKey(0)
  cv2.destroyAllWindows()

cfg = get_cfg()
cfg.OUTPUT_DIR = 'detectron2/output'
cfg.CUDA = 'cuda:0'
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.DATASETS.TRAIN = ('car-project',)
cfg.DATASETS.TEST = () 
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300   # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []  # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.RETINANET.NUM_CLASSES = 1
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 14
cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((14,1),dtype=float).tolist()
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()
