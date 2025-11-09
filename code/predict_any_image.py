import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import os
from datetime import datetime

print("=" * 70)
print("🤖 E-WASTE AI CLASSIFIER - PREDICT ANY IMAGE")
print("=" * 70)

# Load the trained model
print("\n📂 Loading AI model...")
model = keras.models.load_model('../models/best_transfer_model.h5')
print("✅ Model loaded successfully!")

# Category names
categories = ['Battery', 'Keyboard', 'Microwave', 'Mobile', 'Mouse', 
              'PCB', 'Player', 'Printer', 'Television', 'Washing Machine']

# Recycling tips for each category
recycling_tips = {
    'Battery': '⚠️ Contains toxic materials. Take to designated battery recycling center.',
    'Keyboard': '♻️ Contains plastic and small electronic parts. Recycle at e-waste facility.',
    'Microwave': '⚡ Contains hazardous components. Requires special disposal at e-waste center.',
    'Mobile': '📱 Contains precious metals. Can be refurbished or recycled at mobile stores.',
    'Mouse': '🖱️ Contains plastic and small electronics. Recycle at e-waste facility.',
    'PCB': '🔌 Contains valuable metals and toxic materials. Take to certified e-waste recycler.',
    'Player': '🎵 Contains electronics and plastic. Recycle at e-waste facility.',
    'Printer': '🖨️ Contains plastic, metal, and ink cartridges. Take to e-waste center.',
    'Television': '📺 Contains hazardous materials. Requires professional e-waste disposal.',
    'Washing Machine': '🧺 Large appliance. Schedule pickup with e-waste recycling service.'
}

def predict_image(image_path):
    """Predict e-waste category from image"""
    
    # Load and preprocess image
    img = Image.open(image_path)
    img_display = img.copy()
    
    # Resize for model
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    print("\n🔍 Analyzing image...")
    predictions = model.predict(img_array, verbose=0)
    
    # Get results
    predicted_idx = np.argmax(predictions[0])
    predicted_category = categories[predicted_idx]
    confidence = predictions[0][predicted_idx] * 100
    
    # Get top 3 predictions
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    
    return img_display, predicted_category, confidence, predictions[0], top_3_idx

def display_results(img, predicted_category, confidence, all_predictions, top_3_idx):
    """Display prediction results beautifully"""
    
    # Create figure
    fig = plt.figure(figsize=(16, 6))
    
    # Layout: Image | Bar Chart | Info Box
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.5, 1])
    
    # 1. Show original image
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(img)
    ax1.axis('off')
    
    # Add prediction on image
    color = 'green' if confidence > 80 else 'orange' if confidence > 60 else 'red'
    ax1.set_title(f'🎯 Prediction: {predicted_category}\n' + 
                  f'Confidence: {confidence:.1f}%',
                  fontsize=16, fontweight='bold', color=color, pad=20)
    
    # 2. Show confidence bars for all categories
    ax2 = fig.add_subplot(gs[1])
    colors = ['green' if i == np.argmax(all_predictions) else 'lightblue' 
              for i in range(len(categories))]
    bars = ax2.barh(categories, all_predictions * 100, color=colors, edgecolor='navy')
    ax2.set_xlabel('Confidence (%)', fontsize=12, fontweight='bold')
    ax2.set_title('AI Confidence for Each Category', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 100])
    ax2.grid(axis='x', alpha=0.3)
    
    # Highlight top prediction
    bars[np.argmax(all_predictions)].set_edgecolor('darkgreen')
    bars[np.argmax(all_predictions)].set_linewidth(3)
    
    # 3. Info box with details
    ax3 = fig.add_subplot(gs[2])
    ax3.axis('off')
    
    info_text = f"""
╔════════════════════════════╗
║   CLASSIFICATION RESULT    ║
╚════════════════════════════╝

📦 Category:
   {predicted_category}

📊 Confidence:
   {confidence:.2f}%

🏆 Top 3 Predictions:
"""
    
    for i, idx in enumerate(top_3_idx, 1):
        info_text += f"   {i}. {categories[idx]}: {all_predictions[idx]*100:.1f}%\n"
    
    info_text += f"\n♻️ Recycling Tip:\n   {recycling_tips[predicted_category]}"
    
    ax3.text(0.05, 0.95, info_text, 
             fontsize=10, 
             verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save result
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = f'../results/prediction_{timestamp}.png'
    plt.savefig(result_path, dpi=150, bbox_inches='tight')
    print(f"✅ Result saved: {result_path}")
    
    plt.show()

def select_and_predict():
    """Open file dialog and predict selected image"""
    
    print("\n" + "=" * 70)
    print("📁 SELECT AN IMAGE FILE")
    print("=" * 70)
    
    # Hide tkinter window
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    # Open file dialog
    file_path = filedialog.askopenfilename(
        title="Select E-Waste Image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
            ("All files", "*.*")
        ]
    )
    
    if not file_path:
        print("❌ No file selected!")
        return
    
    print(f"\n📸 Selected: {os.path.basename(file_path)}")
    
    # Predict
    img, predicted_category, confidence, all_predictions, top_3_idx = predict_image(file_path)
    
    # Display results
    print("\n" + "=" * 70)
    print("🎯 PREDICTION RESULTS")
    print("=" * 70)
    print(f"\n✅ Category: {predicted_category}")
    print(f"📊 Confidence: {confidence:.2f}%")
    print(f"\n🏆 Top 3 Predictions:")
    for i, idx in enumerate(top_3_idx, 1):
        print(f"   {i}. {categories[idx]}: {all_predictions[idx]*100:.2f}%")
    print(f"\n♻️ Recycling Tip:")
    print(f"   {recycling_tips[predicted_category]}")
    print("\n" + "=" * 70)
    
    # Show visual results
    display_results(img, predicted_category, confidence, all_predictions, top_3_idx)

# Main program
if __name__ == "__main__":
    print("\n💡 This tool will:")
    print("   1. Let you select any e-waste image")
    print("   2. Analyze it with AI (95% accuracy)")
    print("   3. Show detailed results")
    print("   4. Give recycling recommendations")
    
    while True:
        print("\n" + "=" * 70)
        input("Press ENTER to select an image (or Ctrl+C to exit)...")
        
        try:
            select_and_predict()
            
            print("\n🔄 Want to test another image?")
        except KeyboardInterrupt:
            print("\n\n👋 Thanks for using E-Waste AI Classifier!")
            print("=" * 70)
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again with a valid image file.")