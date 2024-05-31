"""

The `base` module defines an abstract base class `BaseImageProcessor`, intended for processing images as a part of a machine learning pipeline, specifically before feeding them into image-based models. It provides a template for creating image processors that handle pre-processing steps such as normalization, resizing, cropping, and resampling of images.

The `BaseImageProcessor` class serves as a foundation for creating custom image processors. It defines an initializer and an abstract method `preprocess`, which must be implemented by the subclasses. The initializer sets up the processor with several configurable parameters that are commonly used in image pre-processing.

Key Attributes:
- `image_mean`: A tuple specifying the mean values for normalizing the image channels.
- `image_std`: A tuple specifying the standard deviation values for normalizing the image channels.
- `size`: A tuple indicating the target size to which the images should be resized.
- `crop_size`: A dictionary specifying the size of the crop to be applied to the images if needed.
- `resample`: An enum from `PILImageResampling` that defines the resampling method to be used when resizing images.
- `rescale_factor`: A float value by which the pixel values will be scaled, useful for changing the range of pixel values.
- `data_format`: An enum from `ChannelDimension` that specifies the ordering of dimensions (channels-first or channels-last) in the processed images.

The abstract method `preprocess` serves as a placeholder for the actual image pre-processing logic that must be developed when the base class is extended. Subclasses are expected to implement this method to process a batch of images and prepare them for input to a model.
"""

from abc import ABC, abstractmethod
from typing import Dict

from transformers.image_processing_utils import get_size_dict
from transformers.image_utils import ChannelDimension, PILImageResampling


class BaseImageProcessor(ABC):
    """
    A base class for preprocessing images, intended for subclassing to create specific image processing behaviors.
    This class defines a structure for image preprocessing by providing common attributes that subclasses can
    utilize or overwrite depending on the requirement. It is an abstract class and cannot be instantiated directly.
    Subclasses must implement the 'preprocess' method.

    Attributes:
        image_mean (tuple of float):
             The mean value for each channel, used for normalizing images.
        image_std (tuple of float):
             The standard deviation for each channel, used for normalizing images.
        size (tuple of int):
             The desired output size of the images after processing.
        crop_size (dict of str:
             int): The size to which the images will be cropped; defaults to 384x384 if not provided.
        resample (PILImageResampling):
             The resampling method used when resizing images.
        rescale_factor (float):
             The factor used when rescaling the pixel values of the images.
        data_format (ChannelDimension):
             Specifies the ordering of the dimensions in the images, either 'channels_first' or 'channels_last'.
            The constructor of the class takes optional parameters for customization and sets default values for each.
            Subclasses that inherit from this base class can provide additional parameters or methods for specific preprocessing needs.

    Raises:
        NotImplementedError:
             If the subclass does not implement the 'preprocess' method.

    """

    def __init__(
        self,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        size=(384, 384),
        crop_size: Dict[str, int] = None,
        resample=PILImageResampling.BICUBIC,
        rescale_factor=1 / 255,
        data_format=ChannelDimension.FIRST,
    ):
        """
        Initializes an image preprocessing pipeline with customizable configurations for mean and standard deviation normalization, resizing, cropping, resampling, and rescaling.

        Args:
            image_mean (tuple of float, optional):
                 Tuple representing the mean value for each channel to be used for normalization. Defaults to (0.5, 0.5, 0.5).
            image_std (tuple of float, optional):
                 Tuple representing the standard deviation for each channel to be used for normalization. Defaults to (0.5, 0.5, 0.5).
            size (tuple of int, optional):
                 Tuple representing the desired output size of the image after resizing. Defaults to (384, 384).
            crop_size (Dict[str, int], optional):
                 Dictionary with keys 'height' and 'width', specifying the desired cropping size. If None, defaults to {'height': 384, 'width': 384}.
            resample (PILImageResampling, optional):
                 The resampling algorithm to use when resizing the image. Defaults to BICUBIC.
            rescale_factor (float, optional):
                 The factor by which to multiply the image data after converting from int to float. Typically used to scale pixel values to a standard range (e.g., 0-1). Defaults to 1 / 255.
            data_format (ChannelDimension, optional):
                 Enum specifying the ordering of dimensions in the image array, e.g., 'channels_first' or 'channels_last'. Defaults to ChannelDimension.FIRST.

        Raises:
            ValueError:
                 If the 'crop_size' parameter is not passed as a dictionary with the required keys.
                This constructor sets up the preprocessing parameters and validates the 'crop_size' parameter, ensuring it has the correct structure and defaults.

        """
        crop_size = (
            crop_size if crop_size is not None else {"height": 384, "width": 384}
        )
        crop_size = get_size_dict(
            crop_size, default_to_square=True, param_name="crop_size"
        )

        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.data_format = data_format
        self.crop_size = crop_size

    @abstractmethod
    def preprocess(self, images):
        """
        Performs preprocessing steps on a batch of images.
        This abstract method should be implemented by subclasses to apply specific preprocessing
        operations to the given images before they are used in further processing or inference.
        It is a mandatory method that subclasses need to override.

        Args:
            images:
                 A list or batch of images to be preprocessed. The format of the images
                is dependent on the specific implementation and should be documented
                by the subclass.

        Returns:
            The method does not return a value but should be implemented to modify the
            images in place or to set a class attribute with the preprocessed images.


        """
        pass
