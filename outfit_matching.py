import cv2
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from PIL import Image

# 1️⃣ Data Preparation: Load & Preprocess Fashion Dataset
def load_fashion_data():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    
    train_images = train_images.reshape(-1, 28, 28, 1) / 255.0  # Normalize
    test_images = test_images.reshape(-1, 28, 28, 1) / 255.0
    
    train_labels = to_categorical(train_labels, num_classes=10)
    test_labels = to_categorical(test_labels, num_classes=10)
    
    return train_images, train_labels, test_images, test_labels

# 2️⃣ Build CNN Model
def build_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 3️⃣ Train & Evaluate Model
def train_model():
    train_images, train_labels, test_images, test_labels = load_fashion_data()
    model = build_cnn_model()
    
    model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels), batch_size=64)
    
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test Accuracy: {test_acc:.2f}")
    
    return model

# 4️⃣ Color Detection using OpenCV
def detect_colors(image_path, clusters=3):
    if not os.path.exists(image_path):
        print(f"❌ Error: File '{image_path}' not found.")
        return None
    
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ Error: OpenCV cannot read '{image_path}'. Check the file format.")
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (100, 100))  # Resize for faster processing
    image = image.reshape((-1, 3))

    kmeans = KMeans(n_clusters=clusters, n_init=10)
    kmeans.fit(image)
    colors = kmeans.cluster_centers_

    plt.figure(figsize=(6, 2))
    plt.axis("off")
    plt.imshow([colors.astype(int)])
    plt.show()
    
    return colors

# 5️⃣ Suggest Matching Outfit (Basic Color Harmony Check)
def suggest_outfit(main_color, wardrobe_colors):
    from colorsys import rgb_to_hsv, hsv_to_rgb

    main_hsv = rgb_to_hsv(*main_color / 255)
    
    suggestions = []
    for color in wardrobe_colors:
        color_hsv = rgb_to_hsv(*color / 255)
        
        # Check if the color is complementary (opposite on the color wheel)
        if abs(main_hsv[0] - color_hsv[0]) > 0.5:
            suggestions.append(color)

    return np.array(suggestions) * 255

# Example Usage:
if _name_ == "_main_":
    model = train_model()  # Train CNN
    
    # Detect colors from "love.jpg"
    colors = detect_colors("love.jpg")  
    
    if colors is not None:
        # Suggest outfits based on color harmony
        matching_colors = suggest_outfit(colors[0], colors)
        print("Suggested Colors for Matching Outfit:", matching_colors)
