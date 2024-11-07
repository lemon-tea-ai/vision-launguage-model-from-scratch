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
    height, width = size[0], size[1]
    images = [
        image.resize((width, height), resample=resample) for image in images
    ]
    images = [np.array(image) for image in images]
    images = (images * rescale_factor).astype(np.float32)

    mean = np.array(image_mean, dtype=images.dtype)
    std = np.array(image_std, dtype=images.dtype)
    images = (images - mean) / std

    images = [image.transpose(2, 0, 1) for image in images]
    return images

class PaliGemmaProcessor:
    """
    Main processor class that handles both image and text processing for PaliGemma model.
    Combines image preprocessing with text tokenization in a single pipeline.
    """

    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        super().__init__()

        self.image_seq_length = num_image_tokens  # Number of image tokens to prepend to text
        self.image_size = image_size  # Target size for image processing

        # Tokenizer described here: https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md#tokenizer
        # Add special image token to tokenizer vocabulary
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        
        # Add special tokens for object detection and segmentation tasks
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]  # These tokens are used for object detection (bounding boxes)
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ]  # These tokens are used for object segmentation
        tokenizer.add_tokens(EXTRA_TOKENS)
        
        # Get the ID for the image token for later use
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        # We will add the BOS and EOS tokens ourselves
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

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
        # Currently only supports single image-text pairs
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images for {len(text)} prompts."

        # Process images into normalized tensor format
        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1 / 255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )
        # Convert the list of numpy arrays to a single numpy array with shape [Batch_Size, Channel, Height, Width]
        pixel_values = np.stack(pixel_values, axis=0)
        # Convert the numpy array to a PyTorch tensor
        pixel_values = torch.tensor(pixel_values)

        # Prepend a `self.image_seq_length` number of image tokens to the prompt
        image_text_tokens = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        # Returns the input_ids and attention_mask as PyTorch tensors
        # Tokenize the text and return:
        # - input_ids: Tensor of token IDs from tokenizer's vocabulary [batch_size, seq_len]
        # - attention_mask: Binary tensor indicating real tokens (1) vs padding (0) [batch_size, seq_len]
        inputs = self.tokenizer(
            image_text_tokens,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
        )

        # Combine processed images and tokenized text into single output dictionary
        return {"pixel_values": pixel_values, **inputs}
