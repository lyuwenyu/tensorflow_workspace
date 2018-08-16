import numpy as np 
import random
import glob
import os
from PIL import Image, ImageDraw
from xml.etree import cElementTree as ET 
import json


classes = ['car', 'cone']

info = dict(zip(classes, [0, 0]))

def main(phase='train'):

    data_dir = '/home/wenyu/workspace/detection/lwy'
    raw_data = []

    for d in [phase]:

        anns = glob.glob(os.path.join(data_dir, d, 'anno', '*.xml'))
        imgs = glob.glob(os.path.join(data_dir, d, 'img', '*.jpg'))

        print(len(anns), len(imgs))

        for ann in anns:


            blob = {'filename': '',
                    'names': [], 
                    'bboxes': [],
                    'height': 0,
                    'width': 0
            }

            with open(ann, 'r') as f:
                data = ET.fromstring(f.read())

                for c in data.getchildren():

                    if c.tag == 'filename':
                        blob['filename'] = os.path.join(data_dir, d, 'img', c.text)

                    elif c.tag == 'size':
                        blob['height'] = int(c.find('height').text)
                        blob['width'] = int(c.find('width').text)

                    elif c.tag == 'object':
                        
                        name = c.find('name').text
                        bbox =[float(x.text) for x in c.find('bndbox').getchildren()]

                        if name in classes:
                            blob['names'] += [name]
                            blob['bboxes'] += [ bbox ]
                            info[name] += 1

            if len(blob['names']) == 0:
                print('---------')
                continue

            # img = Image.open(blob['filename'])
            # draw = ImageDraw.Draw(img)
            # for bbx in blob['bboxes']:
            #     draw.rectangle((bbx[0], bbx[1], bbx[2], bbx[3]), outline='red')
            # img.show()

            raw_data += [blob]

    print(info)
    print(len(raw_data))
    _data = {'raw_data': raw_data, 'classes': classes}

    # with open(f'../data/fisheye_{phase}_raw_data.json', 'w') as f:
    #     json.dump(_data, f)


if __name__ == '__main__':

    main('train')