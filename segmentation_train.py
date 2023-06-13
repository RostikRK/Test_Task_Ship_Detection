import pandas as pd
import os

from utils import crop_image, rle_to_mask, augment_image

os.environ["SM_FRAMEWORK"] = "tf.keras"

import segmentation_models as sm
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


# Load the CSV file into a DataFrame
df = pd.read_csv('path_to_file')

# Group the DataFrame by 'ImageId' to get a list of masks for each image
grouped = df.groupby('ImageId')['EncodedPixels'].apply(list)


aug_images = []
aug_masks = []

# Loop over the grouped DataFrame
for filename, rle_list in grouped.items():
    image_path = 'path_to_folder_with_photo' + filename
    # Crop the image into patches
    crops = crop_image(image_path)
    # Convert the run-length encoding to a binary mask
    mask = rle_to_mask(rle_list)
    # Crop the mask into patches
    mask_crops = crop_image(mask)
    # Loop over the image and mask patches
    for img, msk in zip(crops, mask_crops):
        # Normalize the image patch
        img_norm = img / 255.0
        # Reshape the image patch for prediction
        img_pred = np.expand_dims(img_norm, axis=0)
        # Predict if the image patch contains a ship
        prediction = model.predict(img_pred, verbose=0)
        if prediction >= 0.5:  # If the model predicts a ship
            for _ in range(3):
                aug_img, aug_msk = augment_image(img, msk)
                # Append the augmented image and mask patches to their respective lists
                aug_images.append(aug_img)
                aug_masks.append(aug_msk)

# Convert the lists into numpy arrays for future use
aug_images = np.array(aug_images)
aug_masks = np.array(aug_masks)

BACKBONE = 'efficientnetb4'  # Use EfficientNetB4 as the backbone
preprocess_input = sm.get_preprocessing(BACKBONE)

# Split your data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(aug_images, aug_masks, test_size=0.2, random_state=42)

# Preprocess input
x_train = preprocess_input(x_train)
x_val = preprocess_input(x_val)
y_train = y_train.astype('float32')
y_val = y_val.astype('float32')

# Define model
model = sm.Unet(BACKBONE, encoder_weights='imagenet')
model.compile(
    optimizer=Adam(lr=1e-4),  # Decrease learning rate
    loss=sm.losses.binary_crossentropy + sm.losses.dice_loss,  # Use combination of BCE and Dice loss
    metrics=[sm.metrics.IOUScore()],
)

checkpoint = ModelCheckpoint("best_model_segment_resnet.h5", monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1)  # Increase patience
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)  # Add ReduceLROnPlateau callback

callbacks_list = [checkpoint, early_stopping, reduce_lr]

# Fit model
model.fit(
   x=x_train,
   y=y_train,
   batch_size=16,
   epochs=100,
   validation_data=(x_val, y_val),
   callbacks=callbacks_list
)
