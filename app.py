import torch
from diffusers import StableDiffusionPipeline
from huggingface_hub import login
import os

# -------------------------------
# STEP 1: Hugging Face Login
# -------------------------------
# Set your Hugging Face token as an environment variable:
# export HF_TOKEN=your_token_here

hf_token = os.getenv("HF_TOKEN")
if hf_token is None:
    raise ValueError("HF_TOKEN environment variable not set")

login(token=hf_token)

# -------------------------------
# STEP 2: Load Stable Diffusion
# -------------------------------
model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32
)

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

print(f"Model loaded on {device}")

# -------------------------------
# STEP 3: Text-to-Image Prompts
# -------------------------------

prompts = {
    "zero_shot": """
    An educational infographic explaining Zero-Shot Prompting in AI.
    Show a movie review sentiment classification example with labels
    positive and negative. Clean academic style.
    """,

    "few_shot": """
    An infographic explaining Few-Shot Prompting in AI.
    Show multiple movie review examples with labeled sentiments:
    positive, negative, and neutral. Educational layout.
    """,

    "chain_of_thought": """
    An educational illustration explaining Chain-of-Thought prompting.
    Show step-by-step reasoning for a math word problem with numbered steps.
    """
}

# -------------------------------
# STEP 4: Generate Images
# -------------------------------
os.makedirs("outputs", exist_ok=True)

for name, prompt in prompts.items():
    image = pipe(prompt).images[0]
    image.save(f"outputs/{name}.png")
    print(f"{name}.png generated")

print("All images generated successfully.")
