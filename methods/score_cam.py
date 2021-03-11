import numpy as np
import tensorflow as tf

from .base import SaliencyMap

class ScoreCam(SaliencyMap):

    def get_mask(self, image, last_conv_layer_name):
        """Computes ScoreCam mask for a predicted label.

        Args:
            image (ndarray): Original image
            last_conv_layer_name: Name of the last convolutional layer
                                  used to calculate scoreCAM mask.

        Returns:
            ScoreCAM mask.
        """
        # getting original class idx
        top_pred_index = self.get_top_predicted_idx(image)

        last_conv_layer = self.model.get_layer(last_conv_layer_name)
        last_conv_layer_model = tf.keras.Model(self.model.inputs, last_conv_layer.output)

        last_conv_layer_output = last_conv_layer_model(image)
        last_conv_layer_output = last_conv_layer_output.numpy()[0]

        H, W = image.shape[1], image.shape[2]
        score_saliency_map = np.zeros_like(image)

        num_channels = last_conv_layer_output.shape[-1]
        pbar = tqdm(total=num_channels)
        for channel_idx in range(num_channels):
            pbar.update(1)
            output_channel = last_conv_layer_output[:,:,channel_idx]
            output_channel = tf.expand_dims(output_channel,2)
            output_channel = tf.image.resize(output_channel, [H,W], method='bilinear')
            norm_saliency_map = self.tf_norm(output_channel)

            mask_stacked = [norm_saliency_map.numpy()[:,:,0]]*3
            mask_stacked = np.stack(mask_stacked, -1)
            image_masked = np.expand_dims(mask_stacked, 0) * image

            prob_masked = model.predict(image_masked)[0, top_pred_index]
            score_saliency_map +=  prob_masked * output_channel
        pbar.close()
        score_saliency_map = tf.nn.relu(score_saliency_map)
        return score_saliency_map


    def tf_norm(self, x):
        norm_x = (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x))
        return norm_x
