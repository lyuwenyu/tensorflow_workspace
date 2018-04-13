

import tensorflow as tf 
import glob
import os


imgs_path = glob.glob('../dataset/*.jpg')

lines, labels = [], []
for i, l in enumerate(imgs_path):

    lines += [l]
    labels += [i]

print(lines)
print(labels)


def _read_data_from_disk(pp, ll):
    
    img_raw = tf.read_file(pp)

    img_decoded = tf.image.decode_jpeg(img_raw)

    img_resized = tf.image.resize_images(img_decoded, [224,224])

    return img_resized, ll, ll
    

_lines = tf.constant(lines)
_labels = tf.cast(labels, tf.int32)

dataset = tf.data.Dataset.from_tensor_slices( (_lines, _labels) )
dataset = dataset.map(_read_data_from_disk)
dataset = dataset.shuffle(buffer_size=len(lines)).batch(3).repeat(5)


## ont shot 
iter_data = dataset.make_initializable_iterator()
next_batch = iter_data.get_next()

## 
iterator = dataset.make_one_shot_iterator()
next_elem =iterator.get_next()


print(dataset.output_shapes)
print(dataset.output_types)

with tf.Session() as sess:
    
    # sess.run(iter_data.initializer)



    for i in range(500):

        try:

            # x, y1, y2 = sess.run(next_batch)
            
            x, y1, y2 = sess.run(next_elem)
            print(y1, y2)



        except tf.errors.OutOfRangeError:
            print('end...')
            print(i)
            break



