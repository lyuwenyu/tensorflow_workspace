import tensorflow as tf 
import numpy as np 
from PIL import Image
import os


class Detector(object):

    def __init__(self, model_path):
        ''' init'''
        self.model_path = model_path
        self.detection_graph = tf.Graph()
        

        self._init()
            
    def _init(self, ):
        '''_init'''
        
        with self.detection_graph.as_default():
            od_graph = tf.GraphDef()
            with tf.gfile.GFile(self.model_path, 'rb') as f:
                _serialized_graph = f.read()
                od_graph.ParseFromString(_serialized_graph)
                tf.import_graph_def(od_graph, name='')
            
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')

        self.sess = tf.Session(graph=self.detection_graph)
    

    def run(self, img):
        '''run one pass
            
            n h w c

        '''
        assert isinstance(img, np.ndarray) is True, 'numpy ndarray'
        
        if len(img.shape) == 3:
            _img = np.expand_dims(img, 0)
        
        with self.detection_graph.as_default():

            bboxes, scores, classes, num = self.sess.run(
                [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                feed_dict={self.image_tensor: _img}
            )
        
        return bboxes, scores, classes, num


if __name__ == '__main__':

    detector = Detector('../inference/frozen_inference_graph.pb')

    img = Image.open('../../test_images/image1.jpg')
    img = np.array(img)

    b, s, c, n = detector.run(img) 
    print(b, s, c, n)