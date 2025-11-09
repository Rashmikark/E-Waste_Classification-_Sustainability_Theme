import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Path to your dataset
dataset_path = '../dataset/train'

# Get all category folders
categories = sorted(os.listdir(dataset_path))
print("=" * 50)
print("E-WASTE CLASSIFICATION - DATASET EXPLORATION")
print("=" * 50)
print(f"\nTotal Categories Found: {len(categories)}\n")

# Count images in each category
total_images = 0
category_counts = {}

for category in categories:
    category_path = os.path.join(dataset_path, category)
    if os.path.isdir(category_path):
        image_files = [f for f in os.listdir(category_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        count = len(image_files)
        category_counts[category] = count
        total_images += count
        print(f"📦 {category:20s} : {count:4d} images")

print("\n" + "=" * 50)
print(f"TOTAL TRAINING IMAGES: {total_images}")
print("=" * 50)

# Visualize distribution
plt.figure(figsize=(12, 6))
plt.bar(category_counts.keys(), category_counts.values(), color='skyblue', edgecolor='navy')
plt.xlabel('E-Waste Categories', fontsize=12)
plt.ylabel('Number of Images', fontsize=12)
plt.title('Dataset Distribution - E-Waste Categories', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('../dataset_distribution.png')
print("\n✅ Distribution chart saved as 'dataset_distribution.png'")
plt.show()

# Show sample images from each category
print("\n📸 Displaying sample images from each category...")
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Sample Images from Each E-Waste Category', fontsize=16, fontweight='bold')

for idx, category in enumerate(categories[:10]):
    category_path = os.path.join(dataset_path, category)
    images = [f for f in os.listdir(category_path) 
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if images:
        sample_image_path = os.path.join(category_path, images[0])
        img = Image.open(sample_image_path)
        
        row = idx // 5
        col = idx % 5
        axes[row, col].imshow(img)
        axes[row, col].set_title(category, fontsize=10)
        axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('../sample_images.png')
print("✅ Sample images saved as 'sample_images.png'")
plt.show()

print("\n" + "=" * 50)
print("EXPLORATION COMPLETE! ✅")
print("=" * 50)