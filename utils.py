from tensorflow import keras 
import tensorflow as tf 
import numpy as np
from typing import *


def get_initializer(initializer_range: float = 0.02) -> tf.keras.initializers.TruncatedNormal:
    """
    Creates a `tf.keras.initializers.TruncatedNormal` with the given range.

    Args:
        initializer_range (*float*, defaults to 0.02): Standard deviation of the initializer range.

    Returns:
        `tf.keras.initializers.TruncatedNormal`: The truncated normal initializer.
    """
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


def shape_list(tensor: Union[tf.Tensor, np.ndarray]) -> List[int]:
    """
    Deal with dynamic shape in tensorflow cleanly.

    Args:
        tensor (`tf.Tensor` or `np.ndarray`): The tensor we want the shape of.

    Returns:
        `List[int]`: The shape of the tensor as a list.
    """
    if isinstance(tensor, np.ndarray):
        return list(tensor.shape)

    dynamic = tf.shape(tensor)

    if tensor.shape == tf.TensorShape(None):
        return dynamic

    static = tensor.shape.as_list()

    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

                                                                            