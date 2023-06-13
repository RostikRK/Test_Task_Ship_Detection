import pandas as pd
import os

from utils import crop_image, rle_to_mask

os.environ["SM_FRAMEWORK"] = "tf.keras"

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.mixed_precision import set_global_policy


# Load the CSV file into a DataFrame
df = pd.read_csv('path_to_file')

# Group the DataFrame by 'ImageId' to get a list of masks for each image
grouped = df.groupby('ImageId')['EncodedPixels'].apply(list)


# Initialize empty lists to store the image patches and labels
patch_images = []
labels = []

# Loop over the grouped DataFrame
for filename, rle_list in grouped[:int(len(grouped)*0.012)].items():
    image_path = 'path_to_image_folder' + filename
    # Crop the image into patches
    crops = crop_image(image_path)
    # Convert the run-length encoding to a binary mask
    mask = rle_to_mask(rle_list)
    # Crop the mask into patches
    mask_crops = crop_image(mask)
    # Loop over the image and mask patches
    for img, msk in zip(crops, mask_crops):
        # Label the patch as '1' if it contains a ship (mask is not all zeros), '0' otherwise
        label = 1 if np.any(msk) else 0
        # Append the image patch and its label to their respective lists
        patch_images.append(img)
        labels.append(label)

# Convert the lists into numpy arrays for future use
patch_images = np.array(patch_images)
labels = np.array(labels)

X_train, X_val, y_train, y_val = train_test_split(patch_images, labels, test_size=0.2, random_state=42)


set_global_policy('mixed_float16')

# Define a data generator for on-the-fly normalization
datagen = ImageDataGenerator(rescale=1./255)

# Use the data generator to load the data
train_generator = datagen.flow(X_train, y_train, batch_size=32)
val_generator = datagen.flow(X_val, y_val, batch_size=32)

# Load the pre-trained MobileNetV2 model, excluding the top layer
base_model = MobileNet(weights='imagenet', include_top=False)

# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully-connected layer
x = Dense(512, activation='relu')(x)

# Add a logistic layer for binary classification
# Change the policy for the output layer to 'float32' for stability
predictions = Dense(1, activation='sigmoid', dtype='float32')(x)

# Construct the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Set up the early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Set up the model checkpoint callback to save the best model based on validation accuracy
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)

model.fit(train_generator, validation_data=val_generator, steps_per_epoch=len(X_train) // 32,
          epochs=20, callbacks=[early_stopping, model_checkpoint])