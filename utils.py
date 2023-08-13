from tensorflow import keras 
import tensorflow as tf 
import numpy as np


def get_initializer(initializer_range: float = 0.02) -> tf.keras.initializers.TruncatedNormal:
    """
    Creates a `tf.keras.initializers.TruncatedNormal` with the given range.

    Args:
        initializer_range (*float*, defaults to 0.02): Standard deviation of the initializer range.

    Returns:
        `tf.keras.initializers.TruncatedNormal`: The truncated normal initializer.
    """
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)
                                                                            