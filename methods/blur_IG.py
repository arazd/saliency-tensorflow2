import numpy as np
import tensorflow as tf
from scipy import ndimage

class BlurIG(SaliencyMap):

    def get_mask(self, image, max_sigma=50, num_steps=100, grad_step=0.01, sqrt=False, preprocess=True):
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

        if sqrt:
            sigmas = [math.sqrt(float(i)*max_sigma/float(steps)) for i in range(0, steps+1)]
        else:
            sigmas = [float(i)*max_sigma/float(steps) for i in range(0, steps+1)]
        step_vector_diff = [sigmas[i+1] - sigmas[i] for i in range(0, steps)]

        total_gradients = np.zeros_like(x_value)

        for i in range(steps):
            x_step = gaussian_blur(x_value, sigmas[i])
            x_baseline = gaussian_blur(x_value, sigmas[i] + grad_step)
            gaussian_gradient = (x_baseline - x_step) / grad_step
            gradient_map = self.get_gradients(model, x_step)

            total_gradients += step_vector_diff[i] * np.multiply(gaussian_gradient, gradient_map)

        total_gradients *= -1.0
        return total_gradients



    def gaussian_blur(image, sigma):
        """Returns Gaussian blur filtered 3d (WxHxC) image.
        Args:
        image: 3 dimensional ndarray / input image (W x H x C).
        sigma: Standard deviation for Gaussian blur kernel.
        """
        if sigma == 0:
            image_blurred = image
        else:
            image_blurred = ndimage.gaussian_filter(image[0],
                                                     sigma=[sigma, sigma, 0],
                                                     mode='constant')
            image_blurred = np.expand_dims(image_blurred, axis=0)

        return image_blurred
