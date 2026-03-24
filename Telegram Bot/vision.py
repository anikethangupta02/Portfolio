from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

def describe_image(image_path):
    image = Image.open(image_path).convert("RGB")

  
    inputs = processor(image, return_tensors="pt")

    out = model.generate(
        **inputs,
        max_length=50,
        num_beams=5,
        repetition_penalty=1.2
    )

    caption = processor.decode(out[0], skip_special_tokens=True)

   
    words = caption.lower().split()
    filtered = [w for w in words if len(w) > 3]

    tags = list(dict.fromkeys(filtered))[:5]  

    return caption, tags