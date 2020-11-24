import numpy as np
import tensorflow as tf


class VanillaGradients(SaliencyMap):

    def get_mask(self, image, tensor_format=False):
        """Constructs a Vanilla Gradient Map by computing dy/dx.

        Args:
            image: input image in NHWC format, not batched.
        """
        mask = self.get_gradients(image)
        if not tensor_format:
            mask = mask.numpy()
        return mask
