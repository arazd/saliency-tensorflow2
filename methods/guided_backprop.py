import numpy as np
import tensorflow as tf
from scipy import ndimage

from .base import SaliencyMap

class BlurIG(SaliencyMap):

    def get_mask(self, image, preprocess=True):
        """Computes Integrated Gradients for a predicted label.

        Args:
            image (ndarray): Original image
            top_pred_idx: Predicted label for the input image
            baseline (ndarray): The baseline image to start with for interpolation
            num_steps: Number of interpolation steps between the baseline
                and the input used in the computation of integrated gradients. These
                steps along determine the integral approximation error. By default,
                num_steps is set to 50.

        Returns:
            Integrated gradients w.r.t input image
        """

        @tf.custom_gradient
        def guidedRelu(x):
          def grad(dy):
            return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
          return tf.nn.relu(x), grad

        guided_relu_model = Model(
            inputs = [self.model.inputs],
            outputs = [self.model.outputs]
        )
        layer_dict = [layer for layer in guided_relu_model.layers[1:] if hasattr(layer, 'activation')]
        for layer in layer_dict:
          if layer.activation == tf.keras.activations.relu:
            layer.activation = guidedRelu

        with tf.GradientTape() as tape:
          inputs = tf.cast(image, tf.float32)
          tape.watch(inputs)
          outputs = guided_relu_model(inputs)

        grads = tape.gradient(outputs, inputs)[0]
        return grads
