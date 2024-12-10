# Multi-Modal (Vision) Language Model From Scratch

A implementation of a multi-modal vision-language model using PaLI-GEMMA 3B.

## Learning Path
To get the most out of this repository, follow these steps:

1. **Start with Inference**
   - Run the inference.py script first to see the model in action
   - This will help you understand the expected inputs and outputs

2. **Study the Reference Implementations**
   - Examine these fully implemented files:
     - `vision_transformer_1.py` - Vision encoder implementation
     - `text_image_token_processor_1.py` - Token processing logic
     - `decoder_1.py` - Image&Text decoder architecture

3. **Complete the Exercises**
   - Once you understand the reference code, implement the missing components in:
     - `vision_transformer_0.py`
     - `text_image_token_processor_0.py`
     - `decoder_0.py`
   - These files contain TODO comments to guide your implementation

4. **Study Components in Detail**
   Follow this order to understand the data flow:

   a. **Text & Image Processing**
   - Start with `text_image_token_processor_1.py` to understand input preparation
   - Learn how tokenization and image preprocessing work together

   b. **Vision Pipeline**
   - Study `VisionEmbeddings` for image patch processing
   - Examine `VisionTransformer` for feature extraction

   c. **Projection & Merging**
   - Understand `PaliGemmaMultiModalProjector` for feature alignment
   - Study `_merge_input_ids_with_image_features` for modality combination

   d. **Language Model**
   - Examine `GemmaModel` architecture
   - Focus on attention mechanism in `GemmaAttention`

5. **Key Implementation Details**
   Each component serves a specific purpose:
   - Vision Transformer: Converts images into semantic features
   - Text Processor: Handles tokenization and special token insertion
   - Projector: Aligns vision and text feature spaces
   - Decoder: Combines and processes multimodal information

## Setup

1. Clone the repository:
```bash
git clone https://github.com/lemon-tea-ai/vision-launguage-model-from-scratch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Login to Hugging Face to accept the model license:
```bash
huggingface-cli login
```

4. Install Git LFS and download the model:
```bash
git lfs install
git clone https://huggingface.co/google/paligemma-3b-pt-224
```

5. Run inference:
```bash
python inference.py
```

## Example Output

```
Device in use:  cpu
Loading model
Model loaded in 44.23 seconds
Running inference
Result!!!
this building is the white house
InferenceLatency: 14.26 seconds
Memory Usage: 11.15 GB
```

## Performance Benchmarks

| Hardware | RAM | Inference Latency | Memory Usage |
|----------|-----|-------------------|--------------|
| MacBook Pro M3 | 18GB | 14.26 seconds | 11.15 GB |