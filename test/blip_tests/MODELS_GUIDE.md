# BLIP Models Configuration Guide

## Quick Start

Edit the `model_type` in `inference_test.py`:

```python
model_type = "blip2"  # Choose: "blip_captioning", "blip_vqa", or "blip2"
```

## Model Types Explained

### 1. BLIP Captioning (`blip_captioning`)
- **Model**: `Salesforce/blip-image-captioning-base`
- **Use case**: Simple image captioning
- **Text input**: ❌ Ignored (image-only captioning)
- **Best for**: Basic image descriptions

### 2. BLIP VQA (`blip_vqa`)
- **Model**: `Salesforce/blip-vqa-base`
- **Use case**: Visual Question Answering
- **Text input**: ✅ Accepts questions about images
- **Best for**: Simple Q&A about image content

### 3. BLIP-2 (`blip2`)
- **Models**: 
  - `Salesforce/blip2-opt-2.7b` (default - smaller, faster)
  - `Salesforce/blip2-flan-t5-xl` (better instruction following)
  - `Salesforce/blip2-opt-6.7b` (larger, more capable)
- **Use case**: Advanced text-conditioned generation
- **Text input**: ✅ Excellent text conditioning and instruction following
- **Best for**: Complex prompts, detailed descriptions, conversational AI

## Switching BLIP-2 Models

Uncomment the desired model in the configuration section:

```python
# model_name = "Salesforce/blip2-opt-2.7b"      # Smaller, faster (default)
# model_name = "Salesforce/blip2-flan-t5-xl"    # Better instruction following
# model_name = "Salesforce/blip2-opt-6.7b"      # Larger, more capable
```

## Example Prompts

### For BLIP VQA:
- "What are in the image?"
- "What color are the objects?"
- "How many objects are there?"

### For BLIP-2:
- "Describe this image in detail, including colors, objects, and their arrangement."
- "Question: What type of fruit is shown in the image? Answer:"
- "Analyze the composition and artistic elements in this image."

## Memory Requirements

- **BLIP/BLIP VQA**: ~1-2GB VRAM
- **BLIP-2 OPT-2.7B**: ~6GB VRAM (with float16)
- **BLIP-2 FLAN-T5-XL**: ~4GB VRAM (with float16)
- **BLIP-2 OPT-6.7B**: ~14GB VRAM (with float16)

## Performance Notes

- BLIP-2 models are significantly slower but much more capable
- Use float16 precision for BLIP-2 to reduce memory usage
- BLIP-2 FLAN-T5 variants often have better instruction following
- For production use, consider using smaller models first
