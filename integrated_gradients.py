import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import densenet


class IntegratedGradients(SaliencyMap):

    def get_integrated_gradients(self, image, baseline=None, num_steps=50):
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
        # If baseline is not provided, start with a black image
        # having same size as the input image.
        if baseline is None:
            img_size = image.shape
            baseline = np.zeros(img_size).astype(np.float32)
        else:
            baseline = baseline.astype(np.float32)

        img_input = image
        top_pred_idx = self.get_top_predicted_idx(model, image)
        interpolated_image = [
            baseline + (i / num_steps) * (img_input - baseline)
            for i in range(num_steps + 1)
            ]
        interpolated_image = np.vstack(interpolated_image).astype(np.float32)

        interpolated_image = self.preprocess_input(interpolated_image)

        grads = []
        pbar = tqdm(total=num_steps)
        for i, img in enumerate(interpolated_image):
            pbar.update(1)
            img = tf.expand_dims(img, axis=0)
            grad = self.get_gradients(model, img)
            grads.append(grad[0])
        pbar.close()
        grads = tf.convert_to_tensor(grads, dtype=tf.float32)

        # 4. Approximate the integral using the trapezoidal rule
        grads = (grads[:-1] + grads[1:]) / 2.0
        avg_grads = tf.reduce_mean(grads, axis=0)

        # 5. Calculate integrated gradients and return
        integrated_grads = (img_input - baseline) * avg_grads
        return integrated_grads


    def preprocess_input(image):
        preprocessed = densenet.preprocess_input(image)
        return preprocessed
