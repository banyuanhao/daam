from mmdet.apis import init_detector, inference_detector
from mmengine.visualization import Visualizer
import mmcv
import torch 
import os
import cv2
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

config_file = 'modelpara/det/rtmdet_tiny_8xb32-300e_coco.py'
checkpoint_file = 'modelpara/det/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
img_path = 'dataset/val2017/000000000632.jpg'

model = init_detector(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'
result = inference_detector(model, img_path)

# print(result.pred_instances.labels[0])
# print(result.pred_instances.scores[0])
# print(result.pred_instances.bboxes[0])

image = mmcv.imread(img_path, channel_order='rgb')

visualizer = Visualizer(image=image,save_dir='pics')
visualizer.draw_bboxes(result.pred_instances.bboxes[0])
#visualizer.draw_texts(result.pred_instances.labels[0])
a = visualizer.get_image()
#cv2.imwrite('pics/a.png',a)
visualizer.add_image('demo', a)