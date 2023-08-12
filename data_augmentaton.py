# Reference: https://github.com/google-research/simclr/blob/master/data_util.py

import tensorflow as tf
import random
import numpy as np 

mean=[0.485, 0.456, 0.406],
std_dev=[0.229, 0.224, 0.225],

AUTO = tf.data.experimental.AUTOTUNE


@tf.function
def gaussian_blur(image, kernel_size=23, padding='SAME'):
    sigma = tf.random.uniform((1,))* 1.9 + 0.1

    radius = tf.cast(kernel_size / 2, tf.int32)
    kernel_size = radius * 2 + 1
    x = tf.cast(tf.range(-radius, radius + 1), tf.float32)
    blur_filter = tf.exp(
		-tf.pow(x, 2.0) / (2.0 * tf.pow(tf.cast(sigma, tf.float32), 2.0)))
    blur_filter /= tf.reduce_sum(blur_filter)
	# One vertical and one horizontal filter.
    blur_v = tf.reshape(blur_filter, [kernel_size, 1, 1, 1])
    blur_h = tf.reshape(blur_filter, [1, kernel_size, 1, 1])
    num_channels = tf.shape(image)[-1]
    blur_h = tf.tile(blur_h, [1, 1, num_channels, 1])
    blur_v = tf.tile(blur_v, [1, 1, num_channels, 1])
    expand_batch_dim = image.shape.ndims == 3
    if expand_batch_dim:
        image = tf.expand_dims(image, axis=0)
    blurred = tf.nn.depthwise_conv2d(
		image, blur_h, strides=[1, 1, 1, 1], padding=padding)
    blurred = tf.nn.depthwise_conv2d(
		blurred, blur_v, strides=[1, 1, 1, 1], padding=padding)
    if expand_batch_dim:
        blurred = tf.squeeze(blurred, axis=0)
    return blurred

@tf.function
def color_jitter(x, s=0.5):
    x = tf.image.random_brightness(x, max_delta=0.8*s)
    x = tf.image.random_contrast(x, lower=1-0.8*s, upper=1+0.8*s)
    x = tf.image.random_saturation(x, lower=1-0.8*s, upper=1+0.8*s)
    x = tf.image.random_hue(x, max_delta=0.2*s)
    x = tf.clip_by_value(x, 0, 1)
    return x

@tf.function
def color_drop(x):
    x = tf.image.rgb_to_grayscale(x)
    x = tf.tile(x, [1, 1, 3])
    return x

@tf.function
def solarize(image, threshold=128):
  # For each pixel in the image, select the pixel
  # if the value is less than the threshold.
  # Otherwise, subtract 255 from the pixel.
  return tf.where(image < threshold, image, 255 - image)

@tf.function
def _local_augment(image):
	# Random flips
    image = random_apply(tf.image.flip_left_right, image, p=0.5)
    # Randomly apply transformation (color distortions) with probability p.
    image = random_apply(color_jitter, image, p=0.8)
    # Randomly apply grayscale
    image = random_apply(color_drop, image, p=0.2)
	# Randomly apply gausian blur
    image = random_apply(gaussian_blur, image, p=0.5)
	
    return image

@tf.function
def _global_augment(image, use_solarization, gb_prob=1.0):
	# Random flips
    image = random_apply(tf.image.flip_left_right, image, p=0.5)
    # Randomly apply transformation (color distortions) with probability p.
    image = random_apply(color_jitter, image, p=0.8)
    # Randomly apply grayscale
    image = random_apply(color_drop, image, p=0.2)
	# Randomly apply gausian blur
    image = random_apply(gaussian_blur, image, p=gb_prob)
    # apply solarization
    if use_solarization:
        image = random_apply(solarize, image, p=0.2)
	
    return image

@tf.function
def random_resize_crop(image, min_scale, max_scale, crop_size):
	# Conditional resizing
    if crop_size == 224:
        image_shape = 260
        image = tf.image.resize(image, 
                                size=(image_shape, image_shape), 
                                method=tf.image.ResizeMethod.BICUBIC)
    else:
        image_shape = 160
        image = tf.image.resize(image, 
                                size=(image_shape, image_shape), 
                                method=tf.image.ResizeMethod.BICUBIC)
        
	# Get the crop size for given min and max scale
    size = tf.random.uniform(shape=(1,), minval=min_scale*image_shape,
		maxval=max_scale*image_shape, dtype=tf.float32)
    size = tf.cast(size, tf.int32)[0]
	# Get the crop from the image
    crop = tf.image.random_crop(image, (size,size,3))
    crop_resize = tf.image.resize(crop, (crop_size, crop_size))

    return crop_resize

@tf.function
def _standardize_normalize(image):
    image = image / 255.0
    image -= mean
    image /= std_dev
    return image

@tf.function
def random_apply(func, x, p):
    return tf.cond(
		tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
				tf.cast(p, tf.float32)),
		lambda: func(x),
		lambda: x)

@tf.function
def scale_image(image):
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image

@tf.function
def tie_together(image, label, min_scale, max_scale, crop_size, mode='local', indx=0):
	# Retrieve the image features
	# Scale the pixel values
    image = scale_image(image)
	# Random resized crops
    image = random_resize_crop(image, min_scale,
		max_scale, crop_size)
    
    if mode == "local":
        # Color distortions & Gaussian blur
        image = _local_augment(image)
        
    else:
        image = _global_augment(image, 
                                use_solarization=False if indx==0 else True, 
                                gb_prob=1.0 if indx==0 else 0.1)
        
    image = _standardize_normalize(image)

    return image, label


def get_multires_dataset(dataset,
	size_crops,
	num_crops,
	min_scale,
	max_scale,
    modes=["global", "local"],
	options=None):

    loaders = tuple()

    for i, num_crop in enumerate(num_crops):
        for _ in range(num_crop):
            loader = (
					dataset
					.shuffle(1024)
					.map(lambda x, y: tie_together(x, y, min_scale[i],
						max_scale[i], size_crops[i], mode=modes[i], indx=0), num_parallel_calls=AUTO)
				)
            
            print(loader)
            
            if options!=None:
                loader = loader.with_options(options)
                
            loaders += (loader, )

    return loaders

def shuffle_zipped_output(a,b,c,d,e):
	listify = [a,b,c,d,e]
	random.shuffle(listify)

	return listify[0], listify[1], listify[2], \
		listify[3], listify[4]