import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from app.inference.engine import engine
from PIL import Image, ImageDraw
import io

print("--- TESTING LOCAL OLLAMA LFM3-OMNI ---")
engine.load_model()
print(f"Model connected: {engine.is_loaded}")

# Create test image: Red circle on white background
img = Image.new("RGB", (300, 300), color=(255, 255, 255))
draw = ImageDraw.Draw(img)
draw.ellipse([80, 80, 220, 220], fill=(220, 20, 60))

buf = io.BytesIO()
img.save(buf, format="JPEG")

question = "What shape and color is drawn in this image?"
print(f"\nQuestion: {question}")
print("Running local forward pass...")

answer, tokens = engine.generate(buf.getvalue(), question)

print("\n--- MODEL KNOWLEDGE RESPONSE ---")
print(f"Answer: {answer}")
print(f"Tokens: {tokens}")
