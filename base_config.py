from tensorflow import keras 
import tensorflow as tf 
from ml_collections import ConfigDict 


def get_baseconfig(model_type="deit_tiny_patch16_224",
                  image_size=224,
                  patch_size=16,
                  num_heads=3,
                  num_layers=12,
                  projection_dim=192,
                  init_values=None,
                  dropout_rate=0.0,
                  drop_path_rate=0.0,
                  include_top=True
            ):

    config = ConfigDict()

    # base config (common for all model type)
    config.model_name = model_type
    config.patch_size = patch_size
    config.image_size = image_size
    config.num_patches = pow(config.image_size // config.patch_size, 2)
    config.num_layers = num_layers
    config.num_heads = num_heads
    config.projection_dim = projection_dim
    config.classifier = "token"
    config.input_shape = (config.image_size, config.image_size, 3)
    config.init_values = init_values
    config.drop_path_rate = drop_path_rate
    config.dropout_rate = dropout_rate
    config.initializer_range = 0.02
    config.layer_norm_eps = 1e-5
    config.num_classes = 0
    config.name = config.model_name

    config.n_channels = 3
    config.model_type = config.model_name
    config.mlp_units = [ 4 * config.projection_dim, config.projection_dim]
    config.include_top = True

    return config.lock()