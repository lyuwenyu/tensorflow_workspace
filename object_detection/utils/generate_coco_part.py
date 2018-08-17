import numpy as np 
import os
from PIL import Image
from PIL import ImageDraw
import json
from collections import defaultdict
import random

from pycocotools.coco import COCO
from object_detection.utils import dataset_util
# import tensorflow as tf



# classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck', \
#             'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
#             'bench', 'dog', 'cat']
classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', \
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', \
            'dog', 'cat']

traffic = ['car', 'bus', 'truck']
annimal = ['dog', 'cat']
cycle = ['bicycle', 'motorcycle']

label_map = dict(zip(classes, classes))
for t in traffic:
    label_map[t] = 'car'
for a in annimal:
    label_map[a] = 'annimal'
for c in cycle:
    label_map[c] = 'cycle'

print(label_map)


def main(data='train'):

    dataset = [ data+'2017' ] # train2017
    coco_dir = '/home/wenyu/workspace/dataset/coco'

    cls_num_info = {n: 0 for n in label_map.values()}
    cls_img_info = {n: 0 for n in label_map.values()}

    raw_data = []
    for d in dataset:
        
        coco_ann = os.path.join(coco_dir, 'annotations', f'instances_{d}.json')
        coco = COCO(coco_ann)

        img_ids = []
        for catid in coco.getCatIds(catNms=classes):
            img_ids += coco.getImgIds(catIds=catid)
        img_ids = list(set(img_ids))
        random.shuffle(img_ids)
        print('img_ids len', len(img_ids))

        for imgid in img_ids[:40000]:
            ann_ids = coco.getAnnIds(imgid)
            anns = coco.loadAnns(ann_ids)
            
            imginfo = coco.loadImgs(imgid)[0]
            img_name = os.path.join(coco_dir, 'images', d, imginfo['file_name'])
            img_height = imginfo['height']
            img_width = imginfo['width']

            obj_names, obj_bboxes = [], []

            _names = []
            for ann in anns:
                name = coco.loadCats(ann['category_id'])[0]['name']
                if name in classes:
                    _bbx = ann['bbox']
                    _bbx = [_bbx[0], _bbx[1], _bbx[0] + _bbx[2], _bbx[1] + _bbx[3]]
                    obj_bboxes += [ _bbx ]
                    obj_names += [label_map[name]]
            
                    cls_num_info[label_map[name]] += 1
                    _names += [label_map[name]]

            for _n in _names:
                cls_img_info[_n] += 1

            assert len(obj_bboxes) == len(obj_names), ' '
            if len(obj_names) == 0:
                print('--------')
                break 
            
            # print(obj_names)
            # print(obj_bboxes)
            # print(img_name)
            # print(img_height)
            # print(img_width)

            blob = {'filename': img_name,
                    'names': obj_names, 
                    'bboxes': obj_bboxes,
                    'height': img_height,
                    'width': img_width
            }

            raw_data += [blob]

            # img = Image.open(img_name)
            # draw = ImageDraw.Draw(img)
            # for bbx in obj_bboxes:
            #     draw.rectangle((bbx[0], bbx[1], bbx[0]+bbx[2], bbx[1]+bbx[3]), outline='red')
            # img.show()

    print('cls_img_info: ', cls_img_info)
    print(cls_num_info)

    print('raw_data len: ', len(raw_data))

    _data = {'raw_data': raw_data, 'classes': list(label_map.values())}

    # with open(f'../data/coco_{data}_raw_data.json', 'w') as f:
    #     json.dump(_data, f)


if __name__ == '__main__':

    main('train')
    