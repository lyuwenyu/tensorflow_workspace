import json
from object_detection.utils import dataset_util
import tensorflow as tf

phase = 'val'

with open('../data/coco_{}_raw_data.json'.format(phase), 'r') as f:
    coco_data = json.load(f)

with open('../data/fisheye_{}_raw_data.json'.format(phase), 'r') as f:
    fisheye_data = json.load(f)

# with open('../data/fisheye_augm_raw_data.json', 'r') as f:
#     augm_data = json.load(f)

classes = sorted(list(set(coco_data['classes'] + fisheye_data['classes'])))
raw_data = coco_data['raw_data'] + fisheye_data['raw_data'] # + augm_data['raw_data']

print(len(raw_data))

label_map = dict(zip(classes, range(1, len(classes)+1)))

for k, v in label_map.items():
    print('item {')
    print(f'\tid: {v}')
    print(f"\tname: '{k}'")
    print('}\n')

def create_tf_example(example):

  height = example['height'] # Image height
  width = example['width']  # Image width
  filename = example['filename'].encode() # Filename of the image. Empty if image is not from file

  with tf.gfile.GFile(example['filename'], 'rb') as f:  
    encoded_image_data = f.read() # Encoded image bytes

  image_format = b'jpeg' # b'jpeg' or b'png'

  xmins = [bbx[0]/width for bbx in example['bboxes']] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [bbx[2]/width for bbx in example['bboxes']] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [bbx[1]/height for bbx in example['bboxes']] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [bbx[3]/height for bbx in example['bboxes']] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = [n.encode() for n in example['names']] # List of string class name of bounding box (1 per box)
  classes = [ int(label_map[n]) for n in example['names']] # List of integer class id of bounding box (1 per box)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example



# flags = tf.app.flags
# flags.DEFINE_string('output_path', '../data/merged_train_data.record', 'Path to output TFRecord')
# FLAGS = flags.FLAGS


writer = tf.python_io.TFRecordWriter('../data/merged_{}_data.record'.format(phase))

for i, example in enumerate(raw_data):
    tf_example = create_tf_example(example)
    writer.write(tf_example.SerializeToString())

    if i % 1000 == 0:
        print('---  ', i)
writer.close()


# if __name__ == '__main__':
#     tf.app.run()