import numpy as np
import PIL.Image
import tensorflow as tf
from matplotlib import pyplot as plt


def load_img(path_to_img, max_dim=256):
    '''
    :param path_to_img: Path to image
    :param max_dim: Maximum dimension of image -- used to control memory
    :return: tensor representation of the image
    '''

    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)

    img = img[tf.newaxis, :]
    return img


def tensor_to_numpy_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
    return tensor[0]


def tensor_to_image(tensor):
    return PIL.Image.fromarray(
        tensor_to_numpy_image(tensor)
    )


def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    plt.imshow(image)

    if title:
        plt.title(title)