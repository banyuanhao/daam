from mmdet.apis import init_detector, inference_detector
from mmengine.visualization import Visualizer
import mmcv
import torch 
import os
import cv2
coco_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign',
        'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 
        'snowboard',
        'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard',
        'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 
        'potted_plant', 'bed', 'dining_table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
        'oven', 
        
        'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy_bear', 'hair_drier', 'toothbrush'
    ]
    
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

config_file = 'modelpara/det/gfl_x101-32x4d-dconv-c4-c5_fpn_ms-2x_coco.py'
checkpoint_file = 'modelpara/det/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco_20200630_102002-14a2bf25.pth'

# config_file = 'modelpara/det/gfl_x101-32x4d-dconv-c4-c5_fpn_ms-2x_coco.py'
# checkpoint_file = 'modelpara/det/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco_20200630_102002-14a2bf25.pth'

img_path = 'dataset/val2017/000000105912.jpg'

model = init_detector(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'
print(model.CLASSES)
raise ValueError('stop')
result = inference_detector(model, img_path)

# print(result.pred_instances.labels)
print(result.pred_instances.scores)
# print(result.pred_instances.bboxes[0])

image = mmcv.imread(img_path, channel_order='rgb')

visualizer = Visualizer(image=image,save_dir='pics')
for i in range(10):
    visualizer.draw_bboxes(result.pred_instances.bboxes[i])
    visualizer.draw_texts(coco_classes[int(result.pred_instances.labels[i])],result.pred_instances.bboxes[i][0:2])
a = visualizer.get_image()
cv2.imwrite('pics/a.png',a)
#visualizer.add_image('demo', a)