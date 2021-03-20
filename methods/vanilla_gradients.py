import numpy as np
import tensorflow as tf

from .base import SaliencyMap

class VanillaGradients(SaliencyMap):

    def get_mask(self, image):
        """Constructs a Vanilla Gradient Map by computing dy/dx.

        Args:
            image: input image in NHWC format.
        """
        
        return mask


    def get_smooth_mask(self, image, stdev_spread=0.1, n=30, magnitude=False):
        """Constructs a SmoothGrad Saliency Map by computing dy/dx.

        Args:
            image: input image in NHWC format.
        """
        stdev = stdev_spread * (np.max(image) - np.min(image))
        total_gradients = np.zeros_like(image)

        for i in range(n):
            noise = np.random.normal(0, stdev, image.shape)
            grads = self.get_gradients(image + noise)
            if magnitude:
                grads *= grads
            total_gradients += grads

        return total_gradients / n
