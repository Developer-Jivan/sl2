##11 
##madhe ha pn code add kara 
##just in case 
# AIM: CNN with performance improvement + comparison graphs (CIFAR-10)

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

# 1. Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Flatten labels (important)
y_train = y_train.flatten()
y_test = y_test.flatten()

# 2. Normalize (0–255 → 0–1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. Create TWO models

# --- Model 1: Basic CNN ---
def create_basic_model():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# --- Model 2: Improved CNN ---
def create_improved_model():
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        
        # Block 2
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        
        # Block 3
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        
        layers.Flatten(),
        
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),  # reduce overfitting
        
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# 4. Train models

basic_model = create_basic_model()
improved_model = create_improved_model()

print("Training Basic Model...")
history_basic = basic_model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(x_test, y_test),
    verbose=1
)

print("\nTraining Improved Model...")
history_improved = improved_model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(x_test, y_test),
    verbose=1
)


# 5. Evaluate
basic_loss, basic_acc = basic_model.evaluate(x_test, y_test)
improved_loss, improved_acc = improved_model.evaluate(x_test, y_test)

print("\nBasic Model Accuracy:", basic_acc)
print("Improved Model Accuracy:", improved_acc)


# 6. Plot comparison graphs

plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history_basic.history['accuracy'], label='Basic Train Acc')
plt.plot(history_basic.history['val_accuracy'], label='Basic Val Acc')

plt.plot(history_improved.history['accuracy'], label='Improved Train Acc')
plt.plot(history_improved.history['val_accuracy'], label='Improved Val Acc')

plt.title("Accuracy Comparison")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history_basic.history['loss'], label='Basic Train Loss')
plt.plot(history_basic.history['val_loss'], label='Basic Val Loss')

plt.plot(history_improved.history['loss'], label='Improved Train Loss')
plt.plot(history_improved.history['val_loss'], label='Improved Val Loss')

plt.title("Loss Comparison")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()