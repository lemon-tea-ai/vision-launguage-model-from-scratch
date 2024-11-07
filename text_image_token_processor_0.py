from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch

# Standard normalization values used in many computer vision models
# These values help center the image data around 0 and scale it appropriately
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]  # One value for each RGB channel
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]   # One value for each RGB channel


def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    # Quoting from the blog (https://huggingface.co/blog/paligemma#detailed-inference-process):
    #   The input text is tokenized normally.
    #   A <bos> token is added at the beginning, and an additional newline token (\n) is appended.
    #   This newline token is an essential part of the input prompt the model was trained with, so adding it explicitly ensures it's always there.
    #   The tokenized text is also prefixed with a fixed number of <image> tokens.
    # NOTE: from the paper it looks like the `\n` should be tokenized separately, but in the HF implementation this is not done.
    #       ref to HF implementation: https://github.com/huggingface/transformers/blob/7f79a97399bb52aad8460e1da2f36577d5dccfed/src/transformers/models/paligemma/processing_paligemma.py#L55-L73
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"


def rescale(
    image: np.ndarray, scale: float, dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    TODO: Rescale the image pixel values by multiplying with a scale factor
    1. Multiply the image array by the scale factor (typically 1/255 to convert from [0,255] to [0,1])
    2. Convert the result to the specified dtype (default float32)
    
    Reference implementation in text_image_token_processor_1.py:
    startLine: 23
    endLine: 28
    """
    pass


def resize(
    image: Image,
    size: Tuple[int, int],
    resample: Image.Resampling = None,
    reducing_gap: Optional[int] = None,
) -> np.ndarray:
    """
    TODO: Resize the input PIL Image to the specified dimensions
    1. Extract height and width from size tuple
    2. Use PIL's resize method with the specified resampling method
    3. Return the resized image
    
    Reference implementation in text_image_token_processor_1.py:
    startLine: 31
    endLine: 41
    """
    pass


def normalize(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
) -> np.ndarray:
    """
    TODO: Normalize the image using mean and standard deviation
    1. Convert mean and std to numpy arrays with same dtype as image
    2. Subtract mean from image
    3. Divide by standard deviation
    
    Reference implementation in text_image_token_processor_1.py:
    startLine: 44
    endLine: 52
    """
    pass


def process_images(
    images: List[Image.Image],
    size: Dict[str, int] = None,
    resample: Image.Resampling = None,
    rescale_factor: float = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
) -> List[np.ndarray]:
    """
    TODO: Process a list of images through multiple preprocessing steps
    1. Extract height and width from size dictionary
    2. For each image:
        a. Resize to specified dimensions
        b. Convert to numpy array
        c. Rescale pixel values (typically from [0,255] to [0,1])
        d. Normalize using mean and standard deviation
        e. Rearrange dimensions from [H,W,C] to [C,H,W] format
    
    Reference implementation in text_image_token_processor_1.py:
    startLine: 55
    endLine: 84
    """
    pass


class PaliGemmaProcessor:
    """
    TODO: Implement the processor class that handles both image and text processing
    
    Reference implementation in text_image_token_processor_1.py:
    startLine: 87
    endLine: 175
    """
    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        """
        TODO: Initialize the processor
        1. Call super().__init__()
        2. Store image sequence length and image size
        3. Add special tokens to tokenizer:
            - IMAGE_TOKEN
            - Location tokens (<loc####>)
            - Segmentation tokens (<seg###>)
        4. Get image token ID
        5. Configure tokenizer settings (disable automatic special tokens)
        6. Store tokenizer instance
        """
        pass

    def __call__(
        self,
        text: List[str],
        images: List[Image.Image],
        padding: str = "longest",
        truncation: bool = True,
    ) -> dict:
        """
        TODO: Process both images and text
        1. Validate input (should be one image-text pair)
        2. Process images:
            - Resize, rescale, normalize
            - Convert to tensor format
        3. Prepare text:
            - Add image tokens and special tokens
            - Convert to token IDs
        4. Combine everything into a single dictionary
        
        The final output should contain:
        - pixel_values: Processed image tensor
        - input_ids: Token IDs
        - attention_mask: Mask for padding
        """
        pass
