import os
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode
from detectron2 import model_zoo
import glob
import time


cfg = get_cfg()
cfg.OUTPUT_DIR = 'detectron2/output'
cfg.CUDA = 'cuda:0'
register_coco_instances("car-project", {}, "/home/ms/Downloads/cartest.json", "coco-annotator/datasets/cartest")





metadata = MetadataCatalog.get("car-project")
dataset_dicts = DatasetCatalog.get("car-project")
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
predictor = DefaultPredictor(cfg)

im = cv2.imread("../Pictures/a.jpeg")
outputs = predictor(im)
v = Visualizer(im[:, :, ::-1],metadata=metadata, scale=1, instance_mode=ColorMode.IMAGE_BW)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
img = out.get_image()[:, :, ::-1]

cv2.namedWindow("im", cv2.WINDOW_NORMAL)
cv2.imshow("im", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
