import mmcv
import torch
from mmdet.apis import init_detector, inference_detector
from mmengine.visualization import Visualizer
import cv2
from utils_odfn import coco_classes



image = mmcv.imread(img_path)
visualizer = Visualizer(image=image,save_dir='pics')
visualizer.draw_bboxes(torch.tensor(bbox))
visualizer.draw_texts(coco_classes[category_id],torch.tensor(bbox)[0:2])
a = visualizer.get_image()
cv2.imwrite('pics/a.png',a)
visualizer.add_image('demo', a)