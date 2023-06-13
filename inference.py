import pandas as pd
import os

from segmentation_train import preprocess_input
from utils import dice_score, rle_to_mask

os.environ["SM_FRAMEWORK"] = "tf.keras"


import cv2
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import jaccard_score

# Load the CSV file into a DataFrame
df = pd.read_csv('path_to_file')

# Group the DataFrame by 'ImageId' to get a list of masks for each image
grouped = df.groupby('ImageId')['EncodedPixels'].apply(list)

model = load_model('path_to_model', compile=False)

# Get a list of unique image IDs
image_ids = grouped.index.tolist()
# Calculate the 90% mark
split_index = int(len(image_ids) * 0.90)
# Get the test image IDs
test_image_ids = image_ids[split_index:]


# Path to the images
image_dir = "path_to_images"

# Initialize variables to store the true and predicted masks
true_masks = []
pred_masks = []




# Iterate over the test image IDs
for image_id in test_image_ids:
    # Load the image
    image = cv2.imread(os.path.join(image_dir, image_id))

    # Preprocess the image
    image = cv2.resize(image, (256, 256))  # resize to match the model's expected input size
    image = preprocess_input(image)

    # Reshape the image
    image = np.expand_dims(image, axis=0)

    # Make the prediction
    prediction = model.predict(image, verbose=0)[0]

    # Add the predicted mask to the list
    pred_masks.append(prediction)

    # Get the true mask for this image
    rle_encoded_mask = grouped.loc[image_id]

    # Decode the RLE-encoded mask
    true_mask = rle_to_mask(rle_encoded_mask, (768, 768))  # replace with your function to decode the mask
    true_mask = cv2.resize(true_mask, (256, 256))

    # Add the true mask to the list
    true_masks.append(true_mask)

# convert to a binary mask
binary_pred_masks = np.where(np.concatenate(pred_masks) > 0.5, 1, 0)


iou_score = jaccard_score(np.concatenate(true_masks).ravel(), binary_pred_masks.ravel())

print(f"IOU Score: {iou_score}")

dice = dice_score(np.concatenate(true_masks),  binary_pred_masks.ravel())

print(f"Dice Score: {dice}")
