import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
from datetime import datetime

print("=" * 70)
print("WEEK 2: TRANSFER LEARNING - E-WASTE CLASSIFICATION")
print("=" * 70)

# Set paths
train_path = '../dataset/train'
test_path = '../dataset/test'

# Image parameters
IMG_SIZE = 224  # MobileNetV2 uses 224x224
BATCH_SIZE = 32
EPOCHS = 30

print(f"\n📊 Training Configuration:")
print(f"   Model: MobileNetV2 (Transfer Learning)")
print(f"   Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Epochs: {EPOCHS}")

# Enhanced data augmentation
print("\n🔄 Setting up enhanced data augmentation...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load data
print("\n📂 Loading dataset...")
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

num_classes = train_generator.num_classes
class_names = list(train_generator.class_indices.keys())

print(f"\n✅ Data loaded successfully!")
print(f"   Training samples: {train_generator.samples}")
print(f"   Test samples: {test_generator.samples}")
print(f"   Number of classes: {num_classes}")
print(f"   Classes: {', '.join(class_names)}")

# Build Transfer Learning Model
print("\n🏗️ Building Transfer Learning model with MobileNetV2...")

# Load pre-trained MobileNetV2 (without top layers)
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'  # Pre-trained on ImageNet
)

# Freeze base model layers (we'll fine-tune later)
base_model.trainable = False

print(f"   Base Model: MobileNetV2")
print(f"   Pre-trained weights: ImageNet")
print(f"   Trainable: {base_model.trainable}")

# Build complete model
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n📋 Model Architecture:")
model.summary()

# Callbacks for better training
callbacks = [
    # Save best model
    ModelCheckpoint(
        '../models/best_transfer_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    # Stop if no improvement
    EarlyStopping(
        monitor='val_accuracy',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    # Reduce learning rate when stuck
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# Phase 1: Train with frozen base
print("\n" + "=" * 70)
print("🚀 PHASE 1: Training with frozen base model")
print("=" * 70)
print("💡 This trains only the new top layers (faster, ~30-45 mins)")
print()

history_phase1 = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=callbacks,
    verbose=1
)

# Phase 2: Fine-tune some layers
print("\n" + "=" * 70)
print("🔥 PHASE 2: Fine-tuning (unfreezing last layers)")
print("=" * 70)

# Unfreeze the last 30 layers of base model
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

print(f"   Unfrozen last 30 layers for fine-tuning")

# Recompile with lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("💡 Fine-tuning with lower learning rate (~30-45 mins)")
print()

history_phase2 = model.fit(
    train_generator,
    epochs=20,  # Additional epochs for fine-tuning
    validation_data=test_generator,
    callbacks=callbacks,
    verbose=1
)

# Save final model
final_model_path = '../models/final_transfer_model.h5'
model.save(final_model_path)
print(f"\n✅ Final model saved: {final_model_path}")

# Combine training histories
history_combined = {
    'accuracy': history_phase1.history['accuracy'] + history_phase2.history['accuracy'],
    'val_accuracy': history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy'],
    'loss': history_phase1.history['loss'] + history_phase2.history['loss'],
    'val_loss': history_phase1.history['val_loss'] + history_phase2.history['val_loss']
}

# Plot training history
print("\n📊 Generating training plots...")

plt.figure(figsize=(15, 5))

# Accuracy plot
plt.subplot(1, 3, 1)
plt.plot(history_combined['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history_combined['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.axvline(x=len(history_phase1.history['accuracy']), color='red', linestyle='--', label='Fine-tuning starts')
plt.title('Model Accuracy (Transfer Learning)', fontsize=12, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Loss plot
plt.subplot(1, 3, 2)
plt.plot(history_combined['loss'], label='Training Loss', linewidth=2)
plt.plot(history_combined['val_loss'], label='Validation Loss', linewidth=2)
plt.axvline(x=len(history_phase1.history['loss']), color='red', linestyle='--', label='Fine-tuning starts')
plt.title('Model Loss (Transfer Learning)', fontsize=12, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Comparison bar chart
plt.subplot(1, 3, 3)
final_train_acc = history_combined['accuracy'][-1] * 100
final_val_acc = history_combined['val_accuracy'][-1] * 100
best_val_acc = max(history_combined['val_accuracy']) * 100

categories = ['Final\nTraining', 'Final\nValidation', 'Best\nValidation']
values = [final_train_acc, final_val_acc, best_val_acc]
colors = ['#2ecc71', '#3498db', '#f39c12']

bars = plt.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
plt.title('Final Accuracy Metrics', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy (%)')
plt.ylim([0, 100])

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('../results/week2_training_results.png', dpi=300, bbox_inches='tight')
print("✅ Training plots saved: results/week2_training_results.png")
plt.show()

# Final evaluation on test set
print("\n🧪 Evaluating on test set...")
test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)

# Print final results
print("\n" + "=" * 70)
print("🎉 WEEK 2 TRAINING COMPLETE!")
print("=" * 70)
print(f"\n📊 FINAL RESULTS:")
print(f"   Training Accuracy:   {final_train_acc:.2f}%")
print(f"   Validation Accuracy: {final_val_acc:.2f}%")
print(f"   Test Accuracy:       {test_accuracy * 100:.2f}%")
print(f"   Best Val Accuracy:   {best_val_acc:.2f}%")
print(f"\n📈 IMPROVEMENT FROM WEEK 1:")
week1_acc = 52.08
improvement = best_val_acc - week1_acc
print(f"   Week 1 Accuracy: {week1_acc:.2f}%")
print(f"   Week 2 Accuracy: {best_val_acc:.2f}%")
print(f"   Improvement: +{improvement:.2f}%")
print("\n💾 SAVED FILES:")
print(f"   • {final_model_path}")
print(f"   • ../models/best_transfer_model.h5")
print(f"   • ../results/week2_training_results.png")
print("\n" + "=" * 70)
print("🎯 WEEK 2 PROGRESS: 60% COMPLETE!")
print("=" * 70)
