import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
from typing import * 


class MultiCropWrapper(tf.keras.Model):
class MultiCropWrapper(tf.keras.Model):
    def __init__(self, 
                 backbone: Union[tf.keras.Model, tf.keras.layers.Layer],
                 head: Union[tf.keras.Model, tf.keras.layers.Layer], 
                 **kwargs)-> tf.keras.Model:
      
        super(MultiCropWrapper, self).__init__(**kwargs)
        backbone.head, backbone.fc = tf.identity, tf.identity
        self.head = head
        self.backbone = backbone

    def unique_consecutive(self, x: Union[List, Tuple]) -> np.ndarray:
        shapes = tf.constant([img.shape[1] for img in x], dtype=tf.int32)
        _, _, output = tf.unique_with_counts(
                    shapes,
                    out_idx=tf.dtypes.int32,
                    name=None
                )

        crop_inds = np.cumsum(output.numpy(), 0)

        return crop_inds

    def call(self, x: tf.Tensor) -> tf.Tensor:
        if not isinstance(x, list) and not isinstance(x, tuple):
            x = [x]

        idx_crops = self.unique_consecutive(x)
        start_indx, output = 0, None
        for indx, end_idx in enumerate(idx_crops):
            comb_input = (tf.concat(x[start_indx : end_idx], axis=0))
            _out = self.backbone(comb_input)

            if isinstance(_out, tuple):
                _out = _out[0]

            output = _out if output is None else tf.concat([output, _out], axis=0)

            start_indx = end_idx

        return self.head(output)

    def get_config(self):
      config = super().get_config()
      config["backbone"] = self.backbone
      config['head'] = self.head
      
      return config 

    @classmethod
    def from_config(cls, config):
      return cls(**config)
