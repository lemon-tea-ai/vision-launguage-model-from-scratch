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
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image


def resize(
    image: Image,
    size: Tuple[int, int],
    resample: Image.Resampling = None,
    reducing_gap: Optional[int] = None,
) -> np.ndarray:
    height, width = size
    resized_image = image.resize(
        (width, height), resample=resample, reducing_gap=reducing_gap
    )
    return resized_image


def normalize(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
) -> np.ndarray:
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean) / std
    return image


def process_images(
    images: List[Image.Image],
    size: Dict[str, int] = None,
    resample: Image.Resampling = None,
    rescale_factor: float = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
) -> List[np.ndarray]:
    """
    Main function that processes images through several steps to prepare them for the model:
    1. Resize to specified dimensions
    2. Convert to numpy arrays
    3. Rescale pixel values
    4. Normalize the values
    5. Rearrange dimensions to model's expected format
    """
    height, width = size[0], size[1]
    # Step 1: Resize all images to the same size
    images = [
        resize(image=image, size=(height, width), resample=resample) for image in images
    ]
    # Step 2: Convert PIL images to numpy arrays for numerical processing
    images = [np.array(image) for image in images]
    # Step 3: Rescale pixel values from [0, 255] to [0, 1] range
    images = [rescale(image, scale=rescale_factor) for image in images]
    # Step 4: Normalize images using mean and standard deviation
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]
    # Step 5: Rearrange dimensions from [Height, Width, Channel] to [Channel, Height, Width]
    images = [image.transpose(2, 0, 1) for image in images]
    return images


class PaliGemmaProcessor:
    """
    A processor class that handles both image and text processing for the PaLI-GEMMA model.
    It prepares images and text in the specific format required by the model.
    """
    IMAGE_TOKEN = "<image>"  # Special token that represents image content in the text

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        super().__init__()

        self.image_seq_length = num_image_tokens  # Number of image tokens to use
        self.image_size = image_size  # Target size for image processing

        # Add special tokens that the model understands
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        
        # Add special tokens for object detection and segmentation tasks
        # These tokens help the model understand spatial information in images
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]  # Location tokens for identifying object positions
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ]  # Segmentation tokens for identifying object boundaries
        tokenizer.add_tokens(EXTRA_TOKENS)
        
        # Get the ID for the image token for later use
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        
        # Disable automatic addition of special tokens - we'll handle this manually
        tokenizer.add_bos_token = False  # BOS = Beginning of Sequence
        tokenizer.add_eos_token = False  # EOS = End of Sequence

        self.tokenizer = tokenizer

    def __call__(
        self,
        text: List[str],
        images: List[Image.Image],
        padding: str = "longest",
        truncation: bool = True,
    ) -> dict:
        """
        Main processing function that:
        1. Processes the images into the correct format
        2. Prepares the text with special tokens
        3. Combines everything into the format the model expects
        """
        # Currently only supports processing one image-text pair at a time
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images for {len(text)} prompts."

        # Process the images into tensor format
        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,  # High-quality image resizing method
            rescale_factor=1 / 255.0,  # Convert pixel values from [0, 255] to [0, 1]
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )
        # Stack individual images into a batch
        pixel_values = np.stack(pixel_values, axis=0)
        # Convert to PyTorch tensor for model input
        pixel_values = torch.tensor(pixel_values)

        # Prepare text inputs by adding image tokens and other special tokens
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        # Convert text to token IDs and create attention masks
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",  # Return PyTorch tensors
            padding=padding,      # Pad shorter sequences to match longest one
            truncation=truncation,# Cut off text that's too long
        )

        # Combine image and text inputs into a single dictionary
        return_data = {"pixel_values": pixel_values, **inputs}

        return return_data
