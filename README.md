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