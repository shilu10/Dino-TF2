from tensorflow import keras 
import tensorflow as tf 
import numpy as np 


class DinoHead(tf.keras.models.Model):
    def __init__(
        self,
        in_dim: int = 768,
        out_dim: int = 65536,
        use_bn: bool = False,
        norm_last_layer: bool = True,
        nlayers: int = 3,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        **kwargs
    ):
        super(DinoHead, self).__init__(**kwargs)
        self.in_dim = in_dim
        self.use_bn = use_bn
        self.out_dim = out_dim
        self.nlayers = nlayers
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.norm_last_layer = norm_last_layer
        self.last_layer = tf.keras.layers.Dense(self.out_dim)

        self.mlp_block = self.mlp()

    def mlp(self):
        layer = []
        layer.append(tf.keras.layers.Dense(self.hidden_dim, input_shape=(self.in_dim,)))
        if self.use_bn:
            layer.append(tf.keras.layers.BatchNormalization())
        layer.append(tf.keras.layers.Activation(tf.nn.gelu))
        for _ in range(self.nlayers - 2):
            layer.append(tf.keras.layers.Dense(self.hidden_dim))
        if self.use_bn:
            layer.append(tf.keras.layers.BatchNormalization())
        layer.append(tf.keras.layers.Activation(tf.nn.gelu))
        layer.append(tf.keras.layers.Dense(self.bottleneck_dim))
        return tf.keras.Sequential(layer)

    def call(self, input_tensor, training=False):
        x = self.mlp_block(input_tensor, training)
        x = tf.nn.l2_normalize(x, axis=-1)
        x = self.last_layer(x)
        return x

    def get_config(self):
      config = super().get_config()

      config.update(
          {
              'in_dim' : self.in_dim,
              'use_bn' : self.use_bn,
              'out_dim' : self.out_dim,
              'nlayers' : self.nlayers,
              'hidden_dim' : self.hidden_dim,
              'bottleneck_dim' : self.bottleneck_dim,
              'norm_last_layer' : self.norm_last_layer
          }
        )
      
      return config
    
    @classmethod
    def from_config(cls, config):
      return cls(**config)