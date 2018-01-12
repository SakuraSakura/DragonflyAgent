import numpy as np
from keras import backend as K
import tensorflow as tf
from pspnet.pspnet import PSPNet50
import pspnet.utils

class Segmentation():
    def __init__(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(sess)

        self.pspnet = PSPNet50(nb_classes=150, input_shape=(473, 473),
                               weights="pspnet50_ade20k")

    def segment(self, image):
        class_scores = self.pspnet.predict(image, False)
        class_image = np.argmax(class_scores, axis=2)
        mapped_class_image = pspnet.utils.map_class_id(class_image)
        return mapped_class_image

    def visualize(self, class_image):
        colored_class_image = pspnet.utils.color_class_image(class_image)
        return colored_class_image
