import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import random

print("=" * 60)
print("E-WASTE CLASSIFICATION - TRANSFER LEARNING MODEL TEST")
print("=" * 60)

# Load the trained model (UPDATED TO TRANSFER LEARNING MODEL)
model_path = '../models/best_transfer_model.h5'
print(f"\n📂 Loading model from: {model_path}")
model = keras.models.load_model(model_path)
print("✅ Model loaded successfully!")

# Category names
categories = ['Battery', 'Keyboard', 'Microwave', 'Mobile', 'Mouse', 
              'PCB', 'Player', 'Printer', 'Television', 'Washing Machine']

# Image parameters (UPDATED TO 224 for MobileNetV2)
IMG_SIZE = 224

def predict_image(image_path):
    """Predict e-waste category from image"""
    # Load and preprocess image
    img = Image.open(image_path)
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class] * 100
    
    return predicted_class, confidence, predictions[0]

# Test on random images from test set
test_path = '../dataset/test'
print(f"\n🔍 Testing WEEK 2 model on random images...")

# Select random images from different categories
test_images = []
for category in os.listdir(test_path):
    category_path = os.path.join(test_path, category)
    if os.path.isdir(category_path):
        images = [f for f in os.listdir(category_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if images:
            random_img = random.choice(images)
            test_images.append((os.path.join(category_path, random_img), category))

# Limit to 6 images for display
test_images = random.sample(test_images, min(6, len(test_images)))

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Week 2: Transfer Learning Model - Predictions', fontsize=16, fontweight='bold')

correct_predictions = 0
total_predictions = len(test_images)

print("\n" + "=" * 60)
print("PREDICTION RESULTS:")
print("=" * 60)

for idx, (img_path, true_category) in enumerate(test_images):
    # Predict
    predicted_idx, confidence, all_predictions = predict_image(img_path)
    predicted_category = categories[predicted_idx]
    
    # Check if correct
    is_correct = predicted_category.lower() == true_category.lower()
    if is_correct:
        correct_predictions += 1
    
    # Load image for display
    img = Image.open(img_path)
    
    # Plot
    row = idx // 3
    col = idx % 3
    axes[row, col].imshow(img)
    
    # Title with prediction
    color = 'green' if is_correct else 'red'
    title = f"True: {true_category}\nPredicted: {predicted_category}\nConfidence: {confidence:.1f}%"
    axes[row, col].set_title(title, fontsize=10, color=color, fontweight='bold')
    axes[row, col].axis('off')
    
    # Print result
    status = "✅ CORRECT" if is_correct else "❌ WRONG"
    print(f"\n{status}")
    print(f"   True Label: {true_category}")
    print(f"   Predicted: {predicted_category} ({confidence:.2f}% confidence)")

plt.tight_layout()
plt.savefig('../results/week2_test_predictions.png')
print(f"\n✅ Test results saved as 'results/week2_test_predictions.png'")
plt.show()

# Calculate accuracy
accuracy = (correct_predictions / total_predictions) * 100

print("\n" + "=" * 60)
print("WEEK 2 TEST RESULTS SUMMARY")
print("=" * 60)
print(f"📊 Correct Predictions: {correct_predictions}/{total_predictions}")
print(f"🎯 Test Accuracy: {accuracy:.2f}%")
print("\n📈 COMPARISON:")
print(f"   Week 1 Test: ~33% (2/6)")
print(f"   Week 2 Test: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")
improvement = accuracy - 33
print(f"   Improvement: +{improvement:.2f}%")
print("=" * 60)

# Show top 3 predictions for last image as example
print("\n" + "=" * 60)
print("CONFIDENCE BREAKDOWN (Last Image Example):")
print("=" * 60)
last_pred = all_predictions
top_3_indices = np.argsort(last_pred)[-3:][::-1]
print("\nTop 3 Predictions:")
for i, idx in enumerate(top_3_indices, 1):
    print(f"   {i}. {categories[idx]}: {last_pred[idx]*100:.2f}%")
print("=" * 60)

print("\n✅ Week 2 Testing Complete!")
print("🚀 Ready for Week 3: Beautiful Web Application!")
