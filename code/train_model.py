"""
E-WASTE CLASSIFICATION - CNN MODEL TRAINING
=============================================
Week 1: Building and Training Your AI Model
Progress: 20% → 30%

This script will:
1. Load your e-waste dataset
2. Build a CNN model
3. Train the model
4. Save the trained model
5. Generate training results

Expected Training Time: 1-2 hours
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

print("="*80)
print("🧠 E-WASTE CLASSIFICATION - CNN MODEL TRAINING")
print("="*80)
print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("⏰ Expected Duration: 2-3 hours (60 epochs)")
print("💡 You can do other light work while this runs!")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

print("\n📋 Step 1/7: Configuration")
print("-" * 80)

# Dataset paths - FIXED PATH!
DATASET_PATH = '../dataset/train'  # Your training data folder (fixed!)
IMG_SIZE = 224                      # Image size for CNN (224x224 pixels)
BATCH_SIZE = 32                     # How many images to process at once
EPOCHS = 60                         # How many times to go through all data (INCREASED!)
NUM_CLASSES = 10                    # Number of e-waste categories

# Categories (update if your dataset has different names)
CATEGORIES = [
    'battery', 'computer', 'keyboard', 'mouse', 'printer',
    'washing_machine', 'PCB', 'player', 'microwave', 'mobile'
]

print(f"✅ Dataset Path: {DATASET_PATH}")
print(f"✅ Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"✅ Batch Size: {BATCH_SIZE}")
print(f"✅ Training Epochs: {EPOCHS}")
print(f"✅ Categories: {NUM_CLASSES}")

# Create output folders
os.makedirs('../models', exist_ok=True)
os.makedirs('../results', exist_ok=True)
print("✅ Output folders created")

# ============================================================================
# DATA PREPARATION WITH AUGMENTATION
# ============================================================================

print("\n📊 Step 2/7: Preparing Data with Augmentation")
print("-" * 80)
print("Data augmentation creates variations of your images to help the model learn better!")

# Training data with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,              # Normalize pixels to 0-1
    rotation_range=20,           # Rotate images randomly ±20°
    width_shift_range=0.2,       # Shift horizontally
    height_shift_range=0.2,      # Shift vertically
    shear_range=0.15,            # Shear transformation
    zoom_range=0.2,              # Zoom in/out
    horizontal_flip=True,        # Flip images horizontally
    validation_split=0.2,        # Use 20% for validation
    fill_mode='nearest'
)

# Load training data
print("\n📁 Loading training images...")
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Load validation data
print("📁 Loading validation images...")
validation_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print(f"\n✅ Training samples: {train_generator.samples}")
print(f"✅ Validation samples: {validation_generator.samples}")
print(f"✅ Classes found: {list(train_generator.class_indices.keys())}")

# ============================================================================
# BUILD CNN MODEL
# ============================================================================

print("\n🏗️ Step 3/7: Building CNN Model Architecture")
print("-" * 80)
print("Building a Convolutional Neural Network (CNN) - The AI Brain!")

def build_cnn_model():
    """
    Build a CNN model for e-waste classification
    
    Architecture:
    - 4 Convolutional blocks (learning features)
    - 2 Dense layers (making decisions)
    - Dropout (preventing overfitting)
    """
    
    model = models.Sequential([
        # ============ BLOCK 1: Learn Basic Features ============
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # ============ BLOCK 2: Learn Complex Patterns ============
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # ============ BLOCK 3: Learn Advanced Features ============
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # ============ BLOCK 4: More Complex Features ============
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # ============ CLASSIFIER: Make Final Decision ============
        layers.Flatten(),                        # Convert to 1D
        layers.Dense(512, activation='relu'),    # Decision layer 1
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),    # Decision layer 2
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')  # Final prediction
    ])
    
    return model

# Create the model
model = build_cnn_model()

# Display model summary
print("\n📋 Model Architecture:")
model.summary()

print(f"\n✅ Total parameters: {model.count_params():,}")
print(f"✅ Model size: ~{model.count_params() * 4 / (1024*1024):.1f} MB")

# ============================================================================
# COMPILE MODEL
# ============================================================================

print("\n⚙️ Step 4/7: Compiling Model")
print("-" * 80)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', 
             keras.metrics.Precision(),
             keras.metrics.Recall()]
)

print("✅ Model compiled with:")
print("   • Optimizer: Adam (smart learning algorithm)")
print("   • Loss: Categorical Crossentropy (measures errors)")
print("   • Metrics: Accuracy, Precision, Recall")

# ============================================================================
# SETUP CALLBACKS
# ============================================================================

print("\n🔧 Step 5/7: Setting Up Training Callbacks")
print("-" * 80)

# Save best model
checkpoint = ModelCheckpoint(
    '../models/best_ewaste_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Stop if no improvement
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,  # Increased patience for 60 epochs
    restore_best_weights=True,
    verbose=1
)

# Reduce learning rate if stuck
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,  # Increased patience for 60 epochs
    min_lr=1e-7,
    verbose=1
)

callbacks = [checkpoint, early_stop, reduce_lr]

print("✅ Callbacks configured:")
print("   • ModelCheckpoint: Saves best model automatically")
print("   • EarlyStopping: Stops if no improvement (patience=10)")
print("   • ReduceLROnPlateau: Adjusts learning rate if needed")

# ============================================================================
# TRAIN MODEL
# ============================================================================

print("\n" + "="*80)
print("🚀 Step 6/7: TRAINING STARTING NOW!")
print("="*80)
print("\n💡 WHAT YOU'LL SEE:")
print("   • Each epoch shows loss and accuracy")
print("   • Lower loss = better learning")
print("   • Higher accuracy = better predictions")
print("   • Training takes ~3-4 minutes per epoch")
print("\n⏰ ESTIMATED TIME: 2-3 hours total (60 epochs)")
print("   You can minimize this window and do other work!")
print("\n" + "="*80)

# Start training
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)

# ============================================================================
# SAVE FINAL MODEL
# ============================================================================

print("\n💾 Step 7/7: Saving Model and Results")
print("-" * 80)

# Save final model
model.save('../models/final_ewaste_model.h5')
print("✅ Model saved: models/final_ewaste_model.h5")

# Save training history
import pickle
with open('../models/training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
print("✅ Training history saved: models/training_history.pkl")

# ============================================================================
# VISUALIZE TRAINING RESULTS
# ============================================================================

print("\n📊 Generating Training Visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Accuracy
axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0, 0].set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Loss
axes[0, 1].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0, 1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0, 1].set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Precision
axes[1, 0].plot(history.history['precision'], label='Training Precision', linewidth=2)
axes[1, 0].plot(history.history['val_precision'], label='Validation Precision', linewidth=2)
axes[1, 0].set_title('Model Precision Over Time', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Recall
axes[1, 1].plot(history.history['recall'], label='Training Recall', linewidth=2)
axes[1, 1].plot(history.history['val_recall'], label='Validation Recall', linewidth=2)
axes[1, 1].set_title('Model Recall Over Time', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/training_history.png', dpi=300, bbox_inches='tight')
print("✅ Training graphs saved: results/training_history.png")
plt.close()

# ============================================================================
# FINAL RESULTS SUMMARY
# ============================================================================

print("\n" + "="*80)
print("🎉 FINAL RESULTS SUMMARY")
print("="*80)

# Get final metrics
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

best_val_acc = max(history.history['val_accuracy'])
best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1

print(f"\n📊 TRAINING METRICS:")
print(f"   Training Accuracy:   {final_train_acc*100:.2f}%")
print(f"   Validation Accuracy: {final_val_acc*100:.2f}%")
print(f"   Training Loss:       {final_train_loss:.4f}")
print(f"   Validation Loss:     {final_val_loss:.4f}")

print(f"\n🏆 BEST PERFORMANCE:")
print(f"   Best Validation Accuracy: {best_val_acc*100:.2f}%")
print(f"   Achieved at Epoch: {best_epoch}")

print(f"\n💾 SAVED FILES:")
print(f"   • models/best_ewaste_model.h5 (best model during training)")
print(f"   • models/final_ewaste_model.h5 (final model)")
print(f"   • models/training_history.pkl (training data)")
print(f"   • results/training_history.png (visualization)")

print(f"\n⏰ TRAINING DURATION:")
print(f"   Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print("\n" + "="*80)
print("🎯 WEEK 1 PROGRESS: 35% → 40% COMPLETE!")
print("="*80)

print("\n✅ COMPLETED TASKS:")
print("   1. ✅ Environment setup")
print("   2. ✅ Dataset downloaded and explored")
print("   3. ✅ CNN model built")
print("   4. ✅ Model trained with 60 epochs")
print("   5. ✅ Model saved")

print("\n📋 NEXT STEPS:")
print("   • Test the improved model")
print("   • Compare with previous results")
print("   • If accuracy > 75%, move to Week 2")
print("   • If accuracy < 75%, try Transfer Learning")

print("\n💡 QUICK TEST:")
print("   Run this to test your improved model:")
print("   python test_model.py")

print("\n" + "="*80)
print("🎉 CONGRATULATIONS! Your AI model is trained with 60 epochs!")
print("="*80)
print("\n💪 Take a break - you earned it!")
print("🚀 Come back with the results when training is done!")