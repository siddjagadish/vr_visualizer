import numpy as np
import moviepy.editor as mpy
import image_processing.utils as img_utils


class InterpolationVideoCreator(object):

    def __init__(
            self,
            images,
            fps=30,
    ):
        self.images = np.array([img_utils.tensor_to_numpy_image(x) for x in images])  # Get in numpy format
        self.fps = fps
        self.loadings = np.empty((0, len(self.images)))

    def add_linear_video_segment(
            self,
            start_loadings,
            end_loadings,
            n_seconds=1.
    ):
        # Adds a linear video segment
        n_images = int(n_seconds * self.fps)
        proportions = np.linspace(start_loadings, end_loadings, n_images)
        self.loadings = np.vstack((self.loadings, proportions))

    def stitch_video(self):
        frames = list(np.tensordot(self.loadings, self.images, axes=1))
        return mpy.ImageSequenceClip(list(frames), fps=30)