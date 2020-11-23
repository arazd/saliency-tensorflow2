import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import Model


class Saliency_Map():
    def __init__(self, model):
        """Constructs a Vanilla Gradient Map by computing dy/dx.
        Args:
        model: The TensorFlow model used to evaluate Gradient Map.
        Should take image as input and output probabilities vector.
        """
        self.model = model


    def get_top_predicted_idx(self, img_processed):
        preds = self.model.predict(img_processed)
        top_pred_idx = tf.argmax(preds[0])
        return top_pred_idx



    def get_gradients(self, image):
        """Computes the gradients of outputs w.r.t input image.

        Args:
        img_input: 4D image tensor (NHWC)

        Returns:
        Gradients of the predictions w.r.t img_input
        """
        #images = tf.cast(self.image, tf.float32)
        top_pred_idx = self.get_top_predicted_idx(image)

        with tf.GradientTape() as tape:
            tape.watch(image)
            preds = model(image)
            top_class = preds[:, top_pred_idx]

        grads = tape.gradient(top_class, image)
        return grads


    # normalize gradient to range between 0 and 1
    def norm_grad(self, grad_x):
        abs_grads = tf.math.abs(grad_x)
        grad_max_ = np.max(abs_grads, axis=3)[0]
        arr_min, arr_max  = np.min(grad_max_), np.max(grad_max_)
        normalized_grad = (grad_max_ - arr_min) / (arr_max - arr_min + 1e-18)
        return normalized_grad.reshape(1,grad_x.shape[1],grad_x.shape[2],1)
