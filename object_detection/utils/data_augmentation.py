import numpy as np
import json
from PIL import Image, ImageDraw
from ops import perspective_operation
import os
import random

from multiprocessing import Pool


N_augment_per = 5

def func(data):

    img = Image.open(data['filename'])

    # print(os.path.basename(data['filename']))
    augment_blobs = []

    for i in range(N_augment_per):

        blob = {'filename': '',
                'names': [], 
                'bboxes': [],
                'height': 0,
                'width': 0
        }

        _img, M, _ = perspective_operation([img], magnitude=1.0, skew_type='TILT')
        bbx = np.array(data['bboxes'])
        points = np.vstack([bbx[:, :2], bbx[:, 2:]])
        points = np.hstack([points, np.ones((len(points), 1), dtype=np.float32)])
        points = np.dot(points, M.T)
        points[:, 0] = points[:, 0] / (points[:, -1] + 1e-10)
        points[:, 1] = points[:, 1] / (points[:, -1] + 1e-10)
        points = np.hstack([points[:int(len(points)/2), :2], points[int(len(points)/2):, :2]])
        
        points[:, 0] = np.minimum(np.maximum(0, points[:, 0]), _img.size[0]-1)
        points[:, 1] = np.minimum(np.maximum(0, points[:, 1]), _img.size[1]-1)
        points[:, 2] = np.minimum(np.maximum(0, points[:, 2]), _img.size[0]-1)
        points[:, 3] = np.minimum(np.maximum(0, points[:, 3]), _img.size[1]-1)
        
        areas = (points[:, 3] - points[:, 1]) * (points[:, 2] - points[:, 0])
        index = np.where(areas > 60)[0]
        if len(index) == 0:
            print('no objects...')
            continue

        ll = os.path.join(data_augm_dir, 'dataset', '{}_{:0>2}.jpg'.format(os.path.basename(data['filename'])[:-4], i))
        _img.save(ll)

        blob['filename'] = ll
        blob['names'] = [data['names'][i] for i in index] 
        blob['height'] = _img.size[1]
        blob['width'] = _img.size[0]
        blob['bboxes'] = [tuple(lin) for lin in points[index]]

        assert len(blob['names']) == len(blob['bboxes']), 'wrong in augmentation bbox and names'

        # draw = ImageDraw.Draw(_img)
        # for pt in blob['bboxes']:
        #     draw.rectangle( tuple(pt), outline='red')
        # _img.show()

        # print(blob)
        augment_blobs += [blob]

    # break
    return augment_blobs


# data = {'raw_data': augm_data, 'classes': raw_data['classes']}

# with open('../data/fisheye_augm_raw_data.json', 'w') as f:
#     json.dump(data, f)




if __name__ == '__main__':



    with open('../data/fisheye_train_raw_data.json', 'r') as f:
        raw_data = json.load(f)

    print(raw_data['classes'])
    print(len(raw_data['raw_data']))

    data_augm_dir = '/home/wenyu/workspace/detection/lwy/data_augment'

    random.shuffle(raw_data['raw_data'])

    pool = Pool(16)
    
    res = pool.map(func, raw_data['raw_data'])
    pool.close()
    pool.join()

    data = []
    for d in res:
        data += d
    print(len(data))
        
    _data = {'raw_data': data, 'classes': raw_data['classes']}

    with open('../data/fisheye_augm_raw_data.json', 'w') as f:
        json.dump(_data, f)


