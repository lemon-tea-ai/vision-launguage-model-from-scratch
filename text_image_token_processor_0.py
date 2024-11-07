from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch

# Standard ImageNet normalization constants
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]  # Mean values for each RGB channel
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]   # Standard deviation values for each RGB channel


def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    # Quoting from the blog (https://huggingface.co/blog/paligemma#detailed-inference-process):
    #   The input text is tokenized normally.
    #   A <bos> token is added at the beginning, and an additional newline token (\n) is appended.
    #   This newline token is an essential part of the input prompt the model was trained with, so adding it explicitly ensures it's always there.
    #   The tokenized text is also prefixed with a fixed number of <image> tokens.
    # NOTE: from the paper it looks like the `\n` should be tokenized separately, but in the HF implementation this is not done.
    #       ref to HF implementation: https://github.com/huggingface/transformers/blob/7f79a97399bb52aad8460e1da2f36577d5dccfed/src/transformers/models/paligemma/processing_paligemma.py#L55-L73
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"


def process_images(
    images: List[Image.Image],
    size: Tuple[str, int] = None,
    resample: Image.Resampling = None,
    rescale_factor: float = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
) -> List[np.ndarray]:
    """
    Processes a batch of images through several steps:
    1. Resizes images to target dimensions
    2. Converts PIL images to numpy arrays
    3. Rescales pixel values (typically to [0,1] range)
    4. Normalizes using mean and standard deviation
    5. Rearranges dimensions to channel-first format (CHW)
    """
    pass

class PaliGemmaProcessor:
    """
    Main processor class that handles both image and text processing for PaliGemma model.
    Combines image preprocessing with text tokenization in a single pipeline.
    """

    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        pass

    def __call__(
        self,
        text: List[str],
        images: List[Image.Image],
        padding: str = "longest",
        truncation: bool = True,
    ) -> dict:
        """
        Main processing method that:
        1. Processes images into tensor format
        2. Prepends image tokens to text prompts
        3. Tokenizes the modified prompts
        4. Returns combined image and text tensors
        """
        pass
