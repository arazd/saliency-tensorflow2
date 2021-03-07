import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import densenet

from .base import SaliencyMap

class GradCam(SaliencyMap):

    def get_mask(self, image, last_conv_layer_name, preprocess=True):
        """Computes GradCAM saliency map.

        Args:
            image (ndarray): Original image
            top_pred_idx: Predicted label for the input image
            baseline (ndarray): The baseline image to start with for interpolation
            num_steps: Number of interpolation steps between the baseline
                and the input used in the computation of integrated gradients. These
                steps along determine the integral approximation error. By default,
                num_steps is set to 50.

        Returns:
            GradCAM w.r.t input image
        """

        #last_conv_layer_name = 'conv5_block16_concat'
        last_conv_layer = self.model.get_layer(last_conv_layer_name)
        last_conv_layer_idx = self.model.layers.index(last_conv_layer)
        last_conv_layer_model = tf.keras.Model(self.model.inputs, last_conv_layer.output)

        classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input
        classifier_layers = self.model.layers[last_conv_layer_idx+1:]
        for layer in classifier_layers:
            x = layer(x)
        classifier_model = tf.keras.Model(classifier_input, x)


        with tf.GradientTape() as tape:
            # Compute activations of the last conv layer and make the tape watch it
            last_conv_layer_output = last_conv_layer_model(image)
            tape.watch(last_conv_layer_output)
            # Compute class predictions
            preds = classifier_model(last_conv_layer_output)
            top_pred_index = tf.argmax(preds[0])
            top_class_channel = preds[:, top_pred_index]

        # This is the gradient of the top predicted class with regard to
        # the output feature map of the last conv layer
        grads = tape.gradient(top_class_channel, last_conv_layer_output)


        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        last_conv_layer_output = last_conv_layer_output.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        for i in range(pooled_grads.shape[-1]):
            last_conv_layer_output[:, :, i] *= pooled_grads[i]

        # The channel-wise mean of the resulting feature map
        # is our heatmap of class activation
        heatmap = np.mean(last_conv_layer_output, axis=-1)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

        return heatmap
