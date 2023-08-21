from tensorflow import keras 
import tensorflow as tf 
from collections import * 
import os, math, sys, collections
import numpy as np 
from ml_collections import ConfigDict
from .base_config import get_baseconfig
from typing import * 
from .utils import * 


# deleting the custom object
tf.keras.utils.get_custom_objects().clear()

@keras.saving.register_keras_serializable('my_package')
class TFViTPatchEmbeddings(tf.keras.layers.Layer):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """
    def __init__(self, config: ConfigDict, **kwargs)-> tf.keras.layers.Layer:
        super(TFViTPatchEmbeddings, self).__init__(**kwargs)
        image_size = config.image_size
        patch_size = config.patch_size
        projection_dim = config.projection_dim
        n_channels = config.n_channels

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = ((image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1]))

        # calculation of num of patches
        self.num_patches = num_patches
        self.config = config
        self.image_size = image_size
        self.n_channels = n_channels
        self.projection_dim = projection_dim
        self.patch_size = patch_size

        # patch generator
        self.projection = tf.keras.layers.Conv2D(
                    kernel_size=patch_size,
                    strides=patch_size,
                    data_format="channels_last",
                    filters=projection_dim,
                    padding="valid",
                    use_bias=True,
                    kernel_initializer=get_initializer(self.config.initializer_range),
                    bias_initializer="zeros",
                    name="projection"
                )

    def call(self, 
             x: tf.Tensor, 
             interpolate_pos_encoding: bool = True, 
             training: bool = False) -> tf.Tensor:

        #shape = tf.shape(x)
        #batch_size, height, width, n_channel = shape[0], shape[1], shape[2], shape[3]
        batch_size, height, width, n_channels = shape_list(x)
        num_patches = (width / self.patch_size[0]) * (height / self.patch_size[1])

        if not interpolate_pos_encoding:
           if tf.executing_eagerly():
                if height != self.image_size[0] or width != self.image_size[1]:
                    raise ValueError(
                        f"Input image size ({height}*{width}) doesn't match model"
                        f" ({self.image_size[0]}*{self.image_size[1]})."
                    )

        projection = self.projection(x)
        embeddings = tf.reshape(tensor=projection, shape=(batch_size, int(num_patches), -1))

        return embeddings

    def get_config(self):
      config = super().get_config()
      config.update(
          {
              'num_patches': self.num_patches,
              'image_size': self.image_size,
              'n_channels': self.n_channels, 
              'projection_dim': self.projection_dim, 
              'patch_size': self.patch_size
          }
        )
      
      return config 

    @classmethod
    def from_config(cls, config):
      return cls(**config)
      

# position embed
@keras.saving.register_keras_serializable('my_package')
class TFViTEmbeddings(tf.keras.layers.Layer):
    """
    Construct the CLS token, position and patch embeddings.
    """
    def __init__(self, config: ConfigDict, **kwargs)-> tf.keras.layers.Layer:
        super(TFViTEmbeddings, self).__init__(**kwargs)

        self.patch_embeddings = TFViTPatchEmbeddings(config, name="patch_embedding")
        self.dropout = tf.keras.layers.Dropout(rate=config.dropout_rate)
        self.config = config
        self.dropout_rate = config.dropout_rate

    def build(self, input_shape: tf.TensorShape):
        num_patches = self.patch_embeddings.num_patches
        #patch_size = self.config.patch_size
        #patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        #num_patches = (input_shape[1] // patch_size[0]) * (input_shape[2] // patch_size[1])
        self.cls_token = self.add_weight(
            shape=(1, 1, self.config.projection_dim),
            initializer=get_initializer(self.config.initializer_range),
            trainable=True,
            name="cls_token",
        )

        if "distilled" in self.config.model_name:
          self.dist_token = self.add_weight(
            shape=(1, 1, self.config.projection_dim),
            initializer=get_initializer(self.config.initializer_range),
            trainable=True,
            name="dist_token",
        )
          num_patches += 1

        self.position_embeddings = self.add_weight(
            shape=(1, num_patches + 1, self.config.projection_dim),
            initializer=get_initializer(self.config.initializer_range),
            trainable=True,
            name="position_embeddings",
        )

        super().build(input_shape)

    def interpolate_pos_encoding(self, embeddings, height, width) -> tf.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        batch_size, seq_len, dim = shape_list(embeddings)
        num_patches = seq_len - 1

        _, num_positions, _ = shape_list(self.position_embeddings)
        num_positions -= 1

        if num_patches == num_positions and height == width:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        patch_pos_embed = tf.image.resize(
            images=tf.reshape(
                patch_pos_embed, shape=(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
            ),
            size=(h0, w0),
            method="bicubic",
        )

        shape = shape_list(patch_pos_embed)
        assert h0 == shape[-3] and w0 == shape[-2]
        patch_pos_embed = tf.reshape(tensor=patch_pos_embed, shape=(1, -1, dim))
        return tf.concat(values=(class_pos_embed, patch_pos_embed), axis=1)

    def call(self, 
             x: tf.Tensor, 
             interpolate_pos_encoding: bool = True, 
             training: bool = False) -> tf.Tensor:
        #shape = tf.shape(x)
        #batch_size, height, width, n_channels = shape[0], shape[1], shape[2], shape[3]
        batch_size, height, width, n_channels = shape_list(x)

        patch_embeddings = self.patch_embeddings(x, interpolate_pos_encoding, training)

        # repeating the class token for n batch size
        cls_tokens = tf.tile(self.cls_token, (batch_size, 1, 1))

        if "distilled" in self.config.model_name:
          dist_tokens = tf.tile(self.dist_token, (batch_size, 1, 1))
          if dist_tokens.dtype != patch_embeddings.dtype:
            dist_tokens = tf.cast(dist_tokens, patch_embeddings.dtype)

        if cls_tokens.dtype != patch_embeddings.dtype:
          cls_tokens = tf.cast(cls_tokens, patch_embeddings.dtype)

        # adding the [CLS] token to patch_embeeding
        if 'distilled' in self.config.model_name:
          patch_embeddings = tf.concat([cls_tokens, dist_tokens, patch_embeddings], axis=1)
        else:
          patch_embeddings = tf.concat([cls_tokens, patch_embeddings], axis=1)

        # adding positional embedding to patch_embeddings
        if interpolate_pos_encoding:
            encoded_patches = patch_embeddings + self.interpolate_pos_encoding(patch_embeddings, height, width)
        else:
            encoded_patches = patch_embeddings + self.position_embeddings

        encoded_patches = self.dropout(encoded_patches)

        return encoded_patches

    def get_config(self):
      config = super().get_config()
      config['dropout_rate'] = self.dropout_rate 

      return config 

    @classmethod
    def from_config(cls, config):
      return cls(**config)


def mlp(dropout_rate: float = 0.0, 
        hidden_units: Union[List, Tuple] = [192, 768]):
  
    mlp_block = keras.Sequential(
          [
              tf.keras.layers.Dense(hidden_units[0],
                                   activation=tf.nn.gelu,
                                   bias_initializer=keras.initializers.RandomNormal(stddev=1e-6)),
              tf.keras.layers.Dropout(dropout_rate),
              tf.keras.layers.Dense(hidden_units[1],
                                    bias_initializer=keras.initializers.RandomNormal(stddev=1e-6)),
              tf.keras.layers.Dropout(dropout_rate)
          ]
      )
    return mlp_block


# Referred from: github.com:rwightman/pytorch-image-models.
@keras.saving.register_keras_serializable('my_package')
class LayerScale(tf.keras.layers.Layer):
    def __init__(self, config: ConfigDict, **kwargs)-> tf.keras.layers.Layer:
        super().__init__(**kwargs)
        self.projection_dim = config.projection_dim
       # self.gamma = tf.Variable(
       #     config.init_values * tf.ones((config.projection_dim,)),
       #     name="layer_scale",
       #  )
    
    def build(self, input_shape):
      self.gamma = self.add_weight(
            config.init_values * tf.ones((config.projection_dim,)),
            name="layer_scale",
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return x * self.gamma

    def get_config(self):
      config = super().get_conig()
      config['projection_dim'] = self.projection_dim

      return config 

    @classmethod
    def from_config(cls, config):
      return cls(**config)


# drop_path
@keras.saving.register_keras_serializable('my_package')
class StochasticDepth(tf.keras.layers.Layer):
    def __init__(self, 
                 drop_prop: float = 0.0, 
                 **kwargs)-> tf.keras.layers.Layer:
      
        super(StochasticDepth, self).__init__(**kwargs)
        self.drop_prob = drop_prop

    def call(self, 
             x: tf.Tensor, 
             training: bool = None) -> tf.Tensor:
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x

    def get_config(self):
      config = super().get_config()
      config['drop_prob'] = self.drop_prob 

      return config 

    @classmethod
    def from_config(cls, config):
      return cls(**config)


# self_attention (mult-head self attention)
@keras.saving.register_keras_serializable('my_package')
class TFViTSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config: ConfigDict, **kwargs)-> tf.keras.layers.Layer:
        super(TFViTSelfAttention, self).__init__(**kwargs)

        if config.projection_dim % config.num_heads != 0:
            raise ValueError(
                f"The hidden size ({config.projection_dim}) is not a multiple of the number "
                f"of attention heads ({config.num_heads})"
            )

        self.num_attention_heads = config.num_heads
        self.attention_head_size = int(config.projection_dim / config.num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        self.query = keras.layers.Dense(units=self.all_head_size, 
                                        name="query", 
                                        use_bias=True)
        
        self.key = keras.layers.Dense(units=self.all_head_size, 
                                      name="key", 
                                      use_bias=True)
        
        self.value = keras.layers.Dense(units=self.all_head_size, 
                                        name="value", 
                                        use_bias=True)
        
        self.dropout = keras.layers.Dropout(rate=config.dropout_rate)

    def transpose_for_scores(
        self, tensor: tf.Tensor, batch_size: int
    ) -> tf.Tensor:
        # Reshape from [batch_size, seq_length, all_head_size] to [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(
            tensor=tensor,
            shape=(
                batch_size,
                -1,
                self.num_attention_heads,
                self.attention_head_size,
            ),
        )

        # Transpose the tensor from [batch_size, seq_length, num_attention_heads, attention_head_size] to [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor = None,
        output_attentions: bool = False,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        batch_size = tf.shape(hidden_states)[0]
        mixed_query_layer = self.query(inputs=hidden_states)
        mixed_key_layer = self.key(inputs=hidden_states)
        mixed_value_layer = self.value(inputs=hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # (batch size, num_heads, seq_len_q, seq_len_k)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, dk)

        # Normalize the attention scores to probabilities.
        attention_probs = tf.nn.softmax(logits=attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(
            inputs=attention_probs, training=training
        )

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = tf.multiply(attention_probs, head_mask)

        attention_output = tf.matmul(attention_probs, value_layer)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, all_head_size)
        attention_output = tf.reshape(
            tensor=attention_output, shape=(batch_size, -1, self.all_head_size)
        )
        outputs = (
            (attention_output, attention_probs)
            if output_attentions
            else (attention_output,)
        )

        return outputs

    def get_config(self):
      config = super().get_config()
      config['num_attention_heads'] = self.num_attention_heads
      config['attention_head_size'] = self.attention_head_size
      config['all_head_size'] = self.all_head_size
      config['sqrt_att_head_size'] = self.sqrt_att_head_size

      return config 

    @classmethod
    def from_config(cls, config):
      return cls(**config) 


# self-attention output(projection dense layer)
@keras.saving.register_keras_serializable('my_package')
class TFViTSelfOutput(tf.keras.layers.Layer):
    """
    The residual connection is defined in TFViTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: ConfigDict, **kwargs)-> tf.keras.layers.Layer:
        super(TFViTSelfOutput, self).__init__(**kwargs)

        self.dense = keras.layers.Dense(
            units=config.projection_dim, name="dense"
        )
        self.dropout = keras.layers.Dropout(rate=config.dropout_rate)

        self.projection_dim = config.projection_dim
        self.dropout_rate = config.dropout_rate

    def call(
        self,
        hidden_states: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)

        return hidden_states

    def get_config(self):
      config = super().get_config()
      config['projection_dim'] = self.projection_dim
      config['dropout_rate'] = self.dropout_rate

      return config

    @classmethod
    def from_config(cls, config):
      return cls(**config)


# combine of self_attention and self_output
@keras.saving.register_keras_serializable('my_package')
class TFViTAttention(tf.keras.layers.Layer):
    def __init__(self, config: ConfigDict, **kwargs)-> tf.keras.layers.Layer:
        super(TFViTAttention, self).__init__(**kwargs)

        self.self_attention = TFViTSelfAttention(config, name="attention")
        self.dense_output = TFViTSelfOutput(config, name="output")

    def call(
        self,
        input_tensor: tf.Tensor,
        head_mask: tf.Tensor = None,
        output_attentions: bool = False,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        self_outputs = self.self_attention(
            hidden_states=input_tensor,
            head_mask=head_mask,
            output_attentions=output_attentions,
            training=training,
        )
        attention_output = self.dense_output(
            hidden_states=self_outputs[0]
            if output_attentions
            else self_outputs,
            training=training,
        )
        if output_attentions:
            outputs = (attention_output,) + self_outputs[
                1:
            ]  # add attentions if we output them

        return outputs

    def get_config(self):
      config = super().get_config()

      return config 

    @classmethod
    def from_config(cls, config):
      return cls(**config)


# combining of attention and mlp to create a transformer block
class TFVITTransformerBlock(tf.keras.Model):
    def __init__(self, config: ConfigDict, drop_prob, **kwargs)-> tf.keras.Model:
        super(TFVITTransformerBlock, self).__init__(**kwargs)

        self.attention = TFViTAttention(config)
        #self.mlp = MLP(config, name="mlp_output")
        self.config = config

        self.layernorm_before = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps,
            name="layernorm_before"
            )
        self.layernorm_after = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps,
            name="layernorm_after"
            )

        self.drop_prob = drop_prob
        self.layer_norm_eps = config.layer_norm_eps
        self.mlp_units = config.mlp_units

        self.mlp = mlp(self.config.dropout_rate, self.config.mlp_units)

    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor = False,
        output_attentions: bool = False,
      #  drop_prob: float = 0.0,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:

        # first layernormalization
        x1 = self.layernorm_before(hidden_states)
        attention_output, attention_scores = self.attention(x1, output_attentions=True)

        attention_output = (
                        LayerScale(self.config)(attention_output)
                        if self.config.init_values
                        else attention_output
                      )

        attention_output = (
                    StochasticDepth(self.drop_prob)(attention_output)
                    if self.drop_prob
                    else attention_output
                )

        # first residual connection
        x2 = tf.keras.layers.Add()([attention_output, hidden_states])

        x3 = self.layernorm_after(x2)
        x4 = self.mlp(x3)
        x4 = LayerScale(self.config)(x4) if self.config.init_values else x4
        x4 = StochasticDepth(self.drop_prob)(x4) if self.drop_prob else x4

        # second residual connection
        outputs = tf.keras.layers.Add()([x2, x4])

        if output_attentions:
            return outputs, attention_scores

        return outputs

    def get_config(self):
      config = super().get_config()
      config['drop_prob'] = self.drop_prob
      config['layer_norm_eps'] = self.layer_norm_eps
      config['mlp_units'] = self.mlp_units

      return config

    @classmethod
    def from_config(cls, config):
      return cls(**config)


# vit model
class ViTClassifier(tf.keras.Model):
    """Vision Transformer base class."""

    def __init__(self, config: ConfigDict, **kwargs) -> tf.keras.Model:
        super(ViTClassifier, self).__init__(**kwargs)
        self.config = config
        self.num_classes = config.num_classes

        # Patch embed layer
        self.patch_embed = TFViTEmbeddings(config, name="patch_embedding")
        dpr = [x for x in tf.linspace(0.0, self.config.drop_path_rate, self.config.num_layers)]

        # transformer blocks
        transformer_blocks = [
            TFVITTransformerBlock(config, name=f"transformer_block_{i}", drop_prob=dpr[i])
            for i in range(config.num_layers)
        ]

        self.transformer_blocks = transformer_blocks

        # Other layers.
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)
        self.layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps
        )

    def build(self, input_shape: tf.TensorShape):
      self.head = tf.keras.layers.Dense(
                config.num_classes,
                kernel_initializer="zeros",
                dtype="float32",
                name="classification_head",
            ) if self.num_classes > 0 else tf.identity 

    def forward_blocks(self, 
                       encoded_patches: tf.Tensor) -> tf.Tensor:
      # Initialize a dictionary to store attention scores from each transformer_block.
      attention_scores = dict()

      # Iterate over the number of layers and stack up blocks of Transformer.
      for transformer_module in self.transformer_blocks:
        # Add a Transformer block.
        encoded_patches, attention_score = transformer_module(
                encoded_patches,
                output_attentions = True
            )
        attention_scores[f"{transformer_module.name}_att"] = attention_score

      return encoded_patches

    def call(self, 
             inputs: tf.Tensor, 
             training: bool = None) -> tf.Tensor:
        n = tf.shape(inputs)[0]

        # Create patches and project the patches.
        projected_patches = self.patch_embed(inputs)

        # dropout for projected patches
        encoded_patches = self.dropout(projected_patches)

        # passing the encoded_patches into transformer blocks
        encoded_patches = self.forward_blocks(encoded_patches)

        # Final layer normalization.
        representation = self.layer_norm(encoded_patches)
        
        # accessing the cls_token feature
        output = representation[:, 0]

        return output

    def get_last_selfattention(self, 
                               inputs: tf.Tensor, 
                               training: bool = False) -> tf.Tensor:
      n = tf.shape(inputs)[0]

      # Create patches and project the patches.
      projected_patches = self.patch_embed(inputs)

      # dropout for projected patches
      encoded_patches = self.dropout(projected_patches)

      # Iterate over the number of layers and stack up blocks of Transformer.
      for indx, transformer_module in enumerate(self.transformer_blocks):
        # Add a Transformer block.
        encoded_patches, attention_score = transformer_module(
                encoded_patches,
                output_attentions = True
            )
        
        if indx == len(self.transformer_blocks)-1:
          return attention_score
        

    def get_intermediate_layer(self, 
                               inputs: tf.Tensor, 
                               training: bool = False) -> Dict:
      n = tf.shape(inputs)[0]

      # Create patches and project the patches.
      projected_patches = self.patch_embed(inputs)

      # dropout for projected patches
      encoded_patches = self.dropout(projected_patches)

      attention_scores = dict()
      # Iterate over the number of layers and stack up blocks of Transformer.
      for indx, transformer_module in enumerate(self.transformer_blocks):
        # Add a Transformer block.
        encoded_patches, attention_score = transformer_module(
                encoded_patches,
                output_attentions = True
            )
        
        if indx < len(self.transformer_blocks)-1:
          attention_scores[f"{transformer_module.name}_att"] = attention_score

      return attention_scores



def vit_tiny(patch_size=16, **kwargs):
  config = get_baseconfig(model_type="vit_tiny",
                      image_size=224,
                      patch_size=patch_size,
                      num_heads=3,
                      num_layers=12,
                      projection_dim=192,
                      init_values=None,
                      dropout_rate=0.0,
                      drop_path_rate=0.0,
                      include_top=True)
  
  model = ViTClassifier(config, **kwargs)
  return model


def vit_small(patch_size=16, **kwargs):
  config = get_baseconfig(model_type="vit_small",
                      image_size=224,
                      patch_size=patch_size,
                      num_heads=6,
                      num_layers=12,
                      projection_dim=384,
                      init_values=None,
                      dropout_rate=0.0,
                      drop_path_rate=0.0,
                      include_top=True)
  
  model = ViTClassifier(config, **kwargs)
  return model


def vit_base(patch_size=16, **kwargs):
  config = get_baseconfig(model_type="vit_base",
                      image_size=224,
                      patch_size=patch_size,
                      num_heads=12,
                      num_layers=12,
                      projection_dim=768,
                      init_values=None,
                      dropout_rate=0.0,
                      drop_path_rate=0.0,
                      include_top=True)
  
  model = ViTClassifier(config, **kwargs)
  return model
