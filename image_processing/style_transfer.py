from image_processing.utils import load_img
import tensorflow as tf
import tensorflow_hub as hub


STYLE_TRANSFER_MODEL = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')


def style_transfer(content_image, style_image, content_max_dim=512, style_max_dim=256):
    """
    :param content_image: Path to content image or tensor representing content image
    :param style_image: Path to style image or tensor representing style image
    :param content_max_dim: Maximum dimension for the content image
    :param style_max_size: Maximum dimension for the style image
    :return: tensor of stylized image
    """
    # TODO: Allow for prediction on multiple images -- one at a time right now
    if not tf.is_tensor(content_image):
        content_image = load_img(content_image, max_dim=content_max_dim)

    if not tf.is_tensor(style_image):
        style_image = load_img(style_image, max_dim=style_max_dim)
    stylized_image = STYLE_TRANSFER_MODEL(tf.constant(content_image), tf.constant(style_image))[0]
    return stylized_image
