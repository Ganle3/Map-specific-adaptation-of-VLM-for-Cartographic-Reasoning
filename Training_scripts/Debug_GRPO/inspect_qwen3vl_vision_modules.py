import torch
from transformers import Qwen3VLForConditionalGeneration


MODEL_NAME = "Qwen/Qwen3-VL-8B-Thinking"


print("Loading model...")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto",
)

print("\n=== Vision modules ===")

for name, module in model.named_modules():
    if "visual" in name.lower():
        print(f"{name:100s} {module.__class__.__name__}")