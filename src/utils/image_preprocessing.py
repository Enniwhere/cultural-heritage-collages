import argparse
import os
import random
from pathlib import Path
from PIL import Image, ImageOps

# Suppress DecompressionBombWarning for large images if needed, use with caution
# Image.MAX_IMAGE_PIXELS = None 

def preprocess_images(input_dir: str, output_dir: str, resolution: int, num_samples: int):
    """
    Randomly samples images from input_dir, resizes them to resolution x resolution,
    and saves them to output_dir.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.is_dir():
        print(f"Error: Input directory not found: {input_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory ensured: {output_path}")

    image_files = list(input_path.glob('*.[jJ][pP][gG]')) + \
                  list(input_path.glob('*.[jJ][pP][eE][gG]')) + \
                  list(input_path.glob('*.[pP][nN][gG]')) + \
                  list(input_path.glob('*.[wW][eE][bB][pP]'))
    
    if not image_files:
        print(f"Error: No image files found in {input_path}")
        return

    print(f"Found {len(image_files)} images in {input_path}.")

    if num_samples > len(image_files):
        print(f"Warning: Requested {num_samples} samples, but only {len(image_files)} images found. Using all images.")
        sampled_files = image_files
    else:
        sampled_files = random.sample(image_files, num_samples)
        print(f"Randomly selected {len(sampled_files)} images.")

    processed_count = 0
    for img_path in sampled_files:
        try:
            img = Image.open(img_path).convert("RGB") # Ensure RGB
            
            # Resize while maintaining aspect ratio and padding if necessary, 
            # or just force resize. Force resize seems more common for Dreambooth.
            # Option 1: Force resize (might distort aspect ratio)
            img_resized = img.resize((resolution, resolution), Image.Resampling.LANCZOS)
            
            # Option 2: Resize keeping aspect ratio and add padding (uncomment to use)
            # img.thumbnail((resolution, resolution), Image.Resampling.LANCZOS)
            # delta_w = resolution - img.width
            # delta_h = resolution - img.height
            # padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
            # img_resized = ImageOps.expand(img, padding, fill='white') # Or another fill color

            output_filename = output_path / img_path.name
            img_resized.save(output_filename)
            processed_count += 1
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print(f"Successfully processed and saved {processed_count} images to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample and resize images.")
    parser.add_argument("input_dir", type=str, help="Directory containing original images.")
    parser.add_argument("output_dir", type=str, help="Directory to save resized images.")
    parser.add_argument("--resolution", type=int, default=1024, help="Target square resolution (default: 1024).")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of images to randomly sample (default: 100).")
    
    args = parser.parse_args()

    preprocess_images(args.input_dir, args.output_dir, args.resolution, args.num_samples) 