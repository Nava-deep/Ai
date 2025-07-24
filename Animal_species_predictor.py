# Install required libraries if needed (Colab has most pre-installed)
!pip install tensorflow tensorflow-datasets pillow --quiet

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
import numpy as np
from google.colab import files
from PIL import Image
import matplotlib.pyplot as plt
import io
import requests


# Load and prepare the TensorFlow dataset
(ds_train, ds_val), ds_info = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],
    with_info=True,
    as_supervised=True
)

def preprocess(image, label):
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0
    return image, label

train_ds = ds_train.map(preprocess).batch(32).prefetch(1)
val_ds = ds_val.map(preprocess).batch(32).prefetch(1)

class_names = ds_info.features['label'].names
print(f"Classes: {class_names}")

# Build and train the model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(class_names), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Training model...")
model.fit(train_ds, validation_data=val_ds, epochs=3)
model.save('animal_classifier.h5')

# Fetch and predict on an online image (default: a sample cat photo)
test_img_url = 'https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg'  # Replace with any public image URL (e.g., dog: 'https://upload.wikimedia.org/wikipedia/commons/6/6e/Golden_retriever.jpg')
response = requests.get(test_img_url)
img = Image.open(io.BytesIO(response.content)).resize((224, 224))

img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

preds = model.predict(img_array)[0]
pred_idx = np.argmax(preds)
print(f"Predicted class: {class_names[pred_idx]} (Probability: {preds[pred_idx]:.4f})")

# Optional: Uncomment below to add upload functionality
# from google.colab import files
# print("Upload an image if you prefer (otherwise using URL above):")
# uploaded = files.upload()
# if uploaded:
#     img_file = list(uploaded.keys())[0]
#     img = Image.open(io.BytesIO(uploaded[img_file])).resize((224, 224))
#     img_array = np.array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     preds = model.predict(img_array)[0]
#     pred_idx = np.argmax(preds)
#     print(f"Predicted class: {class_names[pred_idx]} (Probability: {preds[pred_idx]:.4f})")


# Step 1: Extract images and labels from val_ds (validation dataset)
# Unbatch and take a subset for quick evaluation (e.g., 100 images; increase for full set)
test_images = []
test_labels = []
for images, labels in val_ds.take(100 // 32 + 1):  # Adjust take() for more images
    test_images.append(images.numpy())
    test_labels.append(labels.numpy())

# Concatenate into single arrays
test_images = np.concatenate(test_images, axis=0)
test_labels = np.concatenate(test_labels, axis=0)

print(f"Extracted {len(test_images)} test images for evaluation.")

# Step 2: Get predictions from the trained model
pred_probs = model.predict(test_images)
pred_labels = np.argmax(pred_probs, axis=1)

# Step 3: Calculate accuracy for each class
class_indices = {name: idx for idx, name in enumerate(class_names)}  # e.g., {'cat': 0, 'dog': 1}

accuracies = []
correct_counts = []
total_counts = []

for class_name in class_names:
    class_idx = class_indices[class_name]
    class_mask = (test_labels == class_idx)
    class_correct = np.sum(pred_labels[class_mask] == class_idx)
    class_total = np.sum(class_mask)
    
    if class_total > 0:
        class_accuracy = (class_correct / class_total) * 100
    else:
        class_accuracy = 0  # Handle if no samples for a class
    
    accuracies.append(class_accuracy)
    correct_counts.append(class_correct)
    total_counts.append(class_total)
    
    print(f"{class_name} accuracy: {class_accuracy:.2f}% ({class_correct}/{class_total})")

# Step 4: Visualize as a pie chart (share of correct predictions)
plt.figure(figsize=(6, 6))
plt.pie(
    correct_counts, 
    labels=class_names, 
    autopct='%1.1f%%', 
    startangle=90,
    colors=['#59C3C3', '#52489C']  # Colors for cat and dog
)
plt.title('Share of Correct Predictions Per Animal Class')
plt.axis('equal')
plt.show()
